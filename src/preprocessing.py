import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


#This script does all the preprocessing. It fills in the missing values 
#and returns the complete Train and Test data

def name(X_train, X_test):
    X_train['Last_Name'] = X_train['Name'].str.split(' ').str[-1].fillna('Nobody')
    X_test['Last_Name'] = X_train['Name'].str.split(' ').str[-1].fillna('Nobody')
    return X_train, X_test

class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.last_name_homeplanet_mapping = None
        self.cabin_homeplanet_mapping = None
        self.most_common_homeplanet = None
        self.last_name_cabin_mapping = None
        self.deck_stats = None
        self.side_counts = None
        self.median_spending_by_deck = None  # New attribute for spending pattern inference

    def fit(self, X, y=None):
        """
        Learns mappings for filling missing HomePlanet, Cabin, and infers cabin_deck from spending.
        """
        X = X.copy()

        # Define spending-related service columns
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        # Learn Last_Name -> HomePlanet mapping
        self.last_name_homeplanet_mapping = X[
            (X['Last_Name'] != 'Nobody') & 
            (~X['Last_Name'].isna()) & 
            (~X['HomePlanet'].isna())
        ].groupby('Last_Name')['HomePlanet'].first().to_dict()

        # Learn Cabin -> HomePlanet mapping
        self.cabin_homeplanet_mapping = X[
            (~X['Cabin'].isna()) & 
            (~X['HomePlanet'].isna())
        ].groupby('Cabin')['HomePlanet'].first().to_dict()

        # Determine the most common HomePlanet
        self.most_common_homeplanet = X['HomePlanet'].mode()[0] if X['HomePlanet'].isna().sum() > 0 else None

        # Learn Last_Name -> Cabin mapping
        self.last_name_cabin_mapping = X[
            (X['Last_Name'] != 'Nobody') & 
            (~X['Last_Name'].isna()) & 
            (~X['Cabin'].isna())
        ].groupby('Last_Name')['Cabin'].first().to_dict()

        # Split Cabin into components
        X[['cabin_deck', 'cabin_num', 'cabin_side']] = X['Cabin'].str.split('/', expand=True)

        # Convert cabin_num to numeric
        X['cabin_num'] = pd.to_numeric(X['cabin_num'], errors='coerce')

        # Learn cabin number distribution per deck
        self.deck_stats = X.groupby('cabin_deck')['cabin_num'].agg(['min', 'max']).to_dict()

        # Learn cabin side distribution per deck
        self.side_counts = X.groupby(['cabin_deck', 'cabin_side']).size().unstack(fill_value=0).to_dict()

        # Learn median spending by deck for inference
        self.median_spending_by_deck = X.groupby('cabin_deck')[service_columns].median()

        return self
    
    def transform(self, X):
        """
        Applies the learned mappings to fill missing HomePlanet, Cabin, and Cabin Deck in X_test.
        """
        X = X.copy()
        
        # Define spending-related service columns
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        # Fill missing HomePlanet based on Last_Name
        X['HomePlanet'] = X.apply(
            lambda row: self.last_name_homeplanet_mapping.get(row['Last_Name'], row['HomePlanet'])
            if pd.isna(row['HomePlanet']) else row['HomePlanet'], axis=1
        )

        # Fill missing HomePlanet based on Cabin
        X['HomePlanet'] = X.apply(
            lambda row: self.cabin_homeplanet_mapping.get(row['Cabin'], row['HomePlanet'])
            if pd.isna(row['HomePlanet']) else row['HomePlanet'], axis=1
        )

        # Fill remaining missing HomePlanet with the most common one
        if self.most_common_homeplanet:
            X['HomePlanet'].fillna(self.most_common_homeplanet, inplace=True)

        # Fill missing Cabin based on Last_Name
        X['Cabin'] = X.apply(
            lambda row: self.last_name_cabin_mapping.get(row['Last_Name'], row['Cabin'])
            if pd.isna(row['Cabin']) else row['Cabin'], axis=1
        )

        # Split Cabin into components
        X[['cabin_deck', 'cabin_num', 'cabin_side']] = X['Cabin'].str.split('/', expand=True)

        # Convert cabin_num to numeric
        X['cabin_num'] = pd.to_numeric(X['cabin_num'], errors='coerce')

        # **NEW: Infer missing cabin_deck based on spending similarity**
        def infer_cabin_deck(row):
            if pd.isna(row['cabin_deck']):
                distances = {}
                for deck, median_spending in self.median_spending_by_deck.iterrows():
                    distance = np.linalg.norm(row[service_columns].fillna(0) - median_spending.fillna(0))
                    distances[deck] = distance
                return min(distances, key=distances.get)
            return row['cabin_deck']
        
        X['cabin_deck'] = X.apply(infer_cabin_deck, axis=1)

        # Assign random cabin_num within the deck range
        def random_cabin_num(row):
            deck = row['cabin_deck']
            if pd.isna(row['cabin_num']) and deck in self.deck_stats['min']:
                return np.random.randint(self.deck_stats['min'][deck], self.deck_stats['max'][deck] + 1)
            return row['cabin_num']
        
        X['cabin_num'] = X.apply(random_cabin_num, axis=1)

        # Assign minority side to cabins
        def get_minority_side(deck):
            if deck not in self.side_counts:
                return np.random.choice(['S', 'P'])
            return 'S' if self.side_counts[deck].get('S', 0) < self.side_counts[deck].get('P', 0) else 'P'

        X['cabin_side'] = X.apply(
            lambda row: get_minority_side(row['cabin_deck']) if pd.isna(row['cabin_side']) else row['cabin_side'],
            axis=1
        )

        return X

class CryoSleepImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    def fit(self, X, y=None):
        # No fitting needed; this is a rule-based transformation.
        return self

    def transform(self, X):
        """
        Infers missing CryoSleep values based on spending behavior.
        """
        X = X.copy()

        def infer_cryosleep(row):
            spending_values = row[self.spending_columns]
            
            if any(pd.notna(spent) and spent > 0 for spent in spending_values):  # If any spending > 0
                return False
            elif all(spent == 0 for spent in spending_values if pd.notna(spent)):  # If all spending == 0
                return True
            else:  # If any spending is NaN
                nan_count = sum(pd.isna(spent) for spent in spending_values)
                return False if nan_count > 1 else True

        # Apply the function to infer CryoSleep
        X['CryoSleep'] = X.apply(infer_cryosleep, axis=1)
        
        return X

class SpendingImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.group_medians = None  # Placeholder for median group statistics

    def fit(self, X, y=None):
        """
        Learn the median spending values for groups based on ('HomePlanet', 'Destination', 'cabin_deck').
        """
        self.group_medians = X.groupby(['HomePlanet', 'Destination', 'cabin_deck'])[self.service_columns].median()
        return self

    def transform(self, X):
        """
        Impute missing spending values based on the defined steps.
        """
        X = X.copy()

        # **Step 1**: Set spending to 0 for CryoSleep passengers
        for service in self.service_columns:
            X.loc[X['CryoSleep'] == True, service] = X.loc[X['CryoSleep'] == True, service].fillna(0)

        # **Step 2**: Impute when only one spending value is missing
        def impute_single_missing_spending(row):
            missing_count = row[self.service_columns].isna().sum()
            if missing_count == 1:
                non_missing_spendings = row[self.service_columns].dropna()
                if len(non_missing_spendings) > 0:
                    min_spending = non_missing_spendings.min()
                    max_spending = non_missing_spendings.max()
                    # Impute a random value between min and max
                    for service in self.service_columns:
                        if pd.isna(row[service]):
                            row[service] = np.random.uniform(min_spending, max_spending)
            return row

        X = X.apply(impute_single_missing_spending, axis=1)

        # **Step 3**: Impute remaining missing values using group medians
        for service in self.service_columns:
            X[service] = X.apply(
                lambda row: self.group_medians.loc[(row['HomePlanet'], row['Destination'], row['cabin_deck']), service]
                if pd.isna(row[service]) and (row['HomePlanet'], row['Destination'], row['cabin_deck']) in self.group_medians.index
                else row[service],
                axis=1
            )

        return X

class DestinationImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.known_lastname_mapping = None
        self.destination_prob = None

    def fit(self, X, y=None):
        """
        Learn the mappings:
        - Last_Name → Destination (first occurrence per Last_Name)
        - HomePlanet → Destination probability distribution
        """
        # Mapping of Last_Name to first known Destination
        self.known_lastname_mapping = X[
            (X['Last_Name'] != 'Nobody') & 
            (~X['Last_Name'].isna()) & 
            (~X['Destination'].isna())
        ].groupby('Last_Name')['Destination'].first().to_dict()

        # Probability distribution of Destination per HomePlanet
        self.destination_prob = X.groupby('HomePlanet')['Destination'].value_counts(normalize=True).unstack()
        
        return self

    def transform(self, X):
        """
        Impute missing values in the Destination column.
        """
        X = X.copy()

        # **Step 1**: Infer Destination using Last_Name
        def fill_by_lastname(row):
            if pd.isna(row['Destination']) and row['Last_Name'] != 'Nobody':
                return self.known_lastname_mapping.get(row['Last_Name'])
            return row['Destination']

        X['Destination'] = X.apply(fill_by_lastname, axis=1)

        # **Step 2**: Infer Destination using HomePlanet probabilities
        def fill_by_homeplanet(row):
            if pd.isna(row['Destination']) and row['HomePlanet'] in self.destination_prob.index:
                probabilities = self.destination_prob.loc[row['HomePlanet']]
                return np.random.choice(probabilities.index, p=probabilities.values)
            return row['Destination']

        X['Destination'] = X.apply(fill_by_homeplanet, axis=1)

        return X

class KNNImputerVIP(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.pipeline = None
    
    def fit(self, X, y=None):
        """
        Fit the KNN model on the data with known 'VIP' labels
        """
        # Prepare the features for training: include 'cabin_deck' and service columns
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        features = ['cabin_deck'] + service_columns
        
        # Extract rows with known VIP status (target)
        train_df = X[X['VIP'].notna()]
        
        # Prepare the features and target
        X_train = train_df[features]
        y_train = train_df['VIP'].astype('category')  # VIP is categorical
        
        # Define the preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), ['cabin_deck']),  # One-hot encode 'cabin_deck'
                ('num', StandardScaler(), service_columns)  # Scale spending columns
            ]
        )

        # Create the full pipeline: preprocess, then apply KNN
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=self.n_neighbors))
        ])

        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        return self

    def transform(self, X):
        """
        Impute missing 'VIP' labels in rows where 'VIP' is NaN
        """
        # Ensure we're not modifying the original DataFrame
        X_copy = X.copy()
        
        # Extract rows with missing VIP status (target)
        test_df = X_copy[X_copy['VIP'].isna()]
        
        # Prepare the features (same as in fit)
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        features = ['cabin_deck'] + service_columns
        
        # Get the missing VIP rows for prediction
        X_test = test_df[features]
        
        # Predict missing VIP values
        test_df['VIP'] = self.pipeline.predict(X_test)
        
        # Update the original DataFrame with predicted VIP labels
        X_copy.loc[test_df.index, 'VIP'] = test_df['VIP']
        
        return X_copy

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    def fit(self, X, y=None):
        # Nothing to fit
        return self
    
    
    
    def transform(self, X):
        
        def determine_gender(name):
            if pd.isna(name) or not isinstance(name, str) or name.strip() == "":
                return np.random.choice(["Male", "Female"])  # 50/50 random assignment
            
            first_name = name.split()[0]  # Extract first name
            return "Female" if first_name[-1].lower() in "aeiou" else "Male"
    
        # Feature engineering: Create new features or modify existing ones
        X_copy = X.copy()
        
        X_copy['total_spending'] = X_copy[self.spending_columns].sum(axis=1)

        X_copy['Money_spent'] = X_copy['total_spending']>0  # Did someone spent any money
       
        X_copy['Gender'] = X_copy['Name'].apply(determine_gender)  # What is the Gender of a person based on last letter

        return X_copy

class FamilyFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Fit function: Create mappings based on training data
        self.family_mapping = None
        
        # Ensure we don't try to create features from the test set
        if isinstance(X, pd.DataFrame):
            # Step 1: Exclude rows where Last_Name is "Nobody" for family mapping
            df_family = X[X['Last_Name'] != 'Nobody'].copy()
            
            # Step 2: Create a unique family identifier based on Last_Name and Cabin
            df_family['family_id'] = df_family.groupby(['Last_Name', 'Cabin']).ngroup()

            # Step 3: Calculate family size
            family_size = df_family.groupby('family_id')['PassengerId'].transform('count')

            # Step 4: Map family_size and family_id back to the original DataFrame
            df_family = df_family[['PassengerId', 'family_id']].assign(family_size=family_size)

            # Map the family information to the full dataframe
            self.family_mapping = df_family[['PassengerId', 'family_size', 'family_id']]

        return self

    def transform(self, X):
        # Transform function: Apply transformations to both train and test data
        X_copy = X.copy()

        # Merge the family information back to the data (handle test data cases)
        if isinstance(X, pd.DataFrame):
            X_copy = X_copy.merge(
                self.family_mapping,
                on='PassengerId',
                how='left'
            )
        
            # Step 5: Fill NaN values for family_size (for passengers with Last_Name "Nobody")
            X_copy['family_size'] = X_copy['family_size'].fillna(0).astype(int)

            # Step 6: Create the Is_family column
            X_copy['Is_family'] = X_copy['family_size'] > 1
            
            # Step 7: Drop the family_id column
            X_copy = X_copy.drop(columns=['family_id'])

        return X_copy

class DropUnwantedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        # Nothing to fit
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)


age_imputer = SimpleImputer(strategy='median')

service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['cabin_deck']),  # One-hot encode 'cabin_deck'
        ('num', StandardScaler(), service_columns)  # Scale spending columns
    ]
)

pipeline = Pipeline([
    ('data_imputer', DataImputer()),
    ('cryosleep_imputer', CryoSleepImputer()),
    ('spending_imputer', SpendingImputer()),
    ('destination_imputer', DestinationImputer()),
    ('age_imputer', age_imputer),
    ('VIPImputer', KNNImputerVIP())
    # Add more preprocessing steps or a classifier
])

drop_columns = ['PassengerId', 'Cabin', 'Name', 'Last_Name', 'kfold']  # Example columns to drop

drop_column_transformer = DropUnwantedColumns(columns_to_drop=drop_columns)


def processing(X_train, X_test):
    X_train, X_test= name(X_train, X_test)
    # Fit and transform on training data
    X_train = pipeline.fit_transform(X_train)

    # Apply transformations to test data 
    X_test = pipeline.transform(X_test)

    # Apply feature engineering for family-related data
    family_feature_engineer = FamilyFeatureEngineer()
    X_train = family_feature_engineer.fit_transform(X_train)
    X_test = family_feature_engineer.transform(X_test)

    # Apply custom feature engineering for gender and spending
    feature_engineer = FeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train)
    X_test = feature_engineer.transform(X_test)

    # Drop unwanted columns from both train and test data
    X_train = drop_column_transformer.transform(X_train)
    X_test = drop_column_transformer.transform(X_test)

    return X_train, X_test



