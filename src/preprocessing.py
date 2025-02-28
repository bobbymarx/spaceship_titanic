import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


#This script does all the preprocessing. It fills in the missing values 
#and returns the complete Train and Test data

def name(X):
    X['Last_Name'] = X['Name'].str.split(' ').str[-1].fillna('Nobody')
   
    return X

class HomePlanetImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # No hyperparameters in this simple example,
        # but you could add a verbose flag or column name options here.
        pass
        

    def fit(self, X, y=None):
        # Work on a copy of X to avoid modifying the original data.
        df = X.copy()

        # Create mapping for HomePlanet using Last_Name
        # Exclude rows where Last_Name is 'Nobody' or missing, and where HomePlanet is missing.
        self.known_last_name_mapping_ = (
            df[(df['Last_Name'] != 'Nobody') &
               (~df['Last_Name'].isna()) &
               (~df['HomePlanet'].isna())]
            .groupby('Last_Name')['HomePlanet']
            .first()
            .to_dict()
        )
        
        # Create mapping for HomePlanet using Cabin where HomePlanet is known.
        self.known_cabin_mapping_ = (
            df[(~df['Cabin'].isna()) &
               (~df['HomePlanet'].isna())]
            .groupby('Cabin')['HomePlanet']
            .first()
            .to_dict()
        )
        
        # Determine the most common HomePlanet in the training data.
        # This will be used as the final fallback.
        mode_series = df['HomePlanet'].mode()
        self.most_common_homeplanet_ = mode_series[0] if not mode_series.empty else None
        
        return self

    def transform(self, X):
        # Create a copy so that the original DataFrame is not modified.
        df = X.copy()

        # Step 1: Fill missing HomePlanet based on Last_Name mapping.
        def fill_by_last_name(row):
            if pd.isna(row['HomePlanet']) and pd.notna(row['Last_Name']):
                if row['Last_Name'] != 'Nobody':
                    return self.known_last_name_mapping_.get(row['Last_Name'], row['HomePlanet'])
            return row['HomePlanet']
        
        df['HomePlanet'] = df.apply(fill_by_last_name, axis=1)
        
        # Step 2: Fill remaining missing HomePlanet based on Cabin mapping.
        def fill_by_cabin(row):
            if pd.isna(row['HomePlanet']) and pd.notna(row['Cabin']):
                return self.known_cabin_mapping_.get(row['Cabin'], row['HomePlanet'])
            return row['HomePlanet']
        
        df['HomePlanet'] = df.apply(fill_by_cabin, axis=1)
        
        # Step 3: Fill any still missing HomePlanet values with the most common HomePlanet.
        df['HomePlanet'] = df['HomePlanet'].fillna(self.most_common_homeplanet_)
        
        return df

class CabinImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # You can add a random_state parameter here if reproducibility is needed.
        pass

    def fit(self, X, y=None):
        df = X.copy()
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Step 1: Build a Last_Name -> Cabin mapping (ignoring 'Nobody' and NaN values)
        self.known_lastname_mapping_ = (
            df[(df['Last_Name'] != 'Nobody') & 
               (~df['Last_Name'].isna()) & 
               (~df['Cabin'].isna())]
            .groupby('Last_Name')['Cabin']
            .first()
            .to_dict()
        )
        
        # Fill missing Cabin using Last_Name mapping
        def fill_by_lastname(row):
            if pd.isna(row['Cabin']) and row['Last_Name'] != 'Nobody':
                return self.known_lastname_mapping_.get(row['Last_Name'])
            return row['Cabin']
        df['Cabin'] = df.apply(fill_by_lastname, axis=1)
        
        # Step 2: Split Cabin into deck, number, and side.
        cabin_split = df['Cabin'].str.split('/', expand=True)
        df['cabin_deck'] = cabin_split[0]
        df['cabin_num'] = cabin_split[1]
        df['cabin_side'] = cabin_split[2]
        
        # Step 3: Compute median spending per cabin_deck (for inferring missing deck)
        self.median_spending_by_deck_ = (
            df.groupby('cabin_deck')[service_columns]
            .median()
        )
        
        # Step 4: Compute deck statistics for cabin_num (min and max per deck)
        df['cabin_num'] = pd.to_numeric(df['cabin_num'], errors='coerce')
        deck_stats = df.groupby('cabin_deck')['cabin_num'].agg(['min', 'max'])
        self.deck_stats_ = deck_stats.to_dict(orient='index')
        
        # Step 5: Compute the minority cabin side per deck
        side_counts = df.groupby(['cabin_deck', 'cabin_side']).size().unstack(fill_value=0)
        minority_side = {}
        for deck in side_counts.index:
            count_s = side_counts.loc[deck].get('S', 0)
            count_p = side_counts.loc[deck].get('P', 0)
            # Choose 'S' if S count is less than P; otherwise 'P'
            minority_side[deck] = 'S' if count_s < count_p else 'P'
        self.minority_side_by_deck_ = minority_side
        
        return self

    def transform(self, X):
        df = X.copy()
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Step 1: Fill Cabin using the learned Last_Name mapping.
        def fill_by_lastname(row):
            if pd.isna(row['Cabin']) and row['Last_Name'] != 'Nobody':
                return self.known_lastname_mapping_.get(row['Last_Name'], row['Cabin'])
            return row['Cabin']
        df['Cabin'] = df.apply(fill_by_lastname, axis=1)
        
        # Step 2: Split Cabin into its components.
        cabin_split = df['Cabin'].str.split('/', expand=True)
        df['cabin_deck'] = cabin_split[0]
        df['cabin_num'] = cabin_split[1]
        df['cabin_side'] = cabin_split[2]
        
        # Step 3: Infer missing cabin_deck based on spending patterns.
        def infer_deck(row):
            if pd.isna(row['cabin_deck']):
                distances = {}
                for deck, medians in self.median_spending_by_deck_.iterrows():
                    passenger_spending = row[service_columns].fillna(0)
                    medians_filled = medians.fillna(0)
                    distances[deck] = np.linalg.norm(passenger_spending - medians_filled)
                if distances:
                    return min(distances, key=distances.get)
            return row['cabin_deck']
        df['cabin_deck'] = df.apply(infer_deck, axis=1)
        
        # Step 4: Fill missing cabin_num by assigning a random number within the deck's range.
        df['cabin_num'] = pd.to_numeric(df['cabin_num'], errors='coerce')
        def assign_cabin_num(row):
            if pd.isna(row['cabin_num']):
                deck = row['cabin_deck']
                stats = self.deck_stats_.get(deck, None)
                if stats is not None and pd.notna(stats.get('min')) and pd.notna(stats.get('max')):
                    return np.random.randint(int(stats.get('min')), int(stats.get('max')) + 1)
            return row['cabin_num']
        df['cabin_num'] = df.apply(assign_cabin_num, axis=1)
        
        # Step 5: Fill missing cabin_side using the minority side per deck.
        def assign_cabin_side(row):
            if pd.isna(row['cabin_side']):
                deck = row['cabin_deck']
                return self.minority_side_by_deck_.get(deck, np.random.choice(['S', 'P']))
            return row['cabin_side']
        df['cabin_side'] = df.apply(assign_cabin_side, axis=1)
        
        return df

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
        X_copy = X.copy()
    
        # Create a mask for rows with missing 'VIP'
        na_vip_mask = X_copy['VIP'].isna()
        
        # Prepare features (same as in fit)
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        features = ['cabin_deck'] + service_columns
        
        # Get the subset of data for prediction
        X_test = X_copy.loc[na_vip_mask, features]
        
        # Predict and assign directly to X_copy using .loc
        X_copy.loc[na_vip_mask, 'VIP'] = self.pipeline.predict(X_test)
        
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
        return X.drop(columns=self.columns_to_drop, errors='ignore')


age_imputer = SimpleImputer(strategy='median')

service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['HomePlanet','Destination','cabin_deck']),  # One-hot encode categorical values
        ('num', StandardScaler(), service_columns),
        ('age_imputer', age_imputer, ['Age']),  # Applies to 'Age' column  # Scale spending columns
        # Map binary/categorical columns to 0/1
        ('binary_vip', OrdinalEncoder(categories=[[False, True]]), ['VIP']),
        ('binary_is_family', OrdinalEncoder(categories=[[False, True]]), ['Is_family']),
        ('binary_is_money', OrdinalEncoder(categories=[[False, True]]), ['Money_spent']),
        ('binary_is_cyro', OrdinalEncoder(categories=[[False, True]]), ['CryoSleep']),
        ('binary_cabin_side', OrdinalEncoder(categories=[['S', 'P']]), ['cabin_side']),
        ('binary_gender', OrdinalEncoder(categories=[['Female', 'Male']]), ['Gender'])
    ],
    remainder='passthrough'
)


pipeline = Pipeline([
    ('home_planet_imputer', HomePlanetImputer()),
    ('Cabin_Imputer', CabinImputer()),
    ('cryosleep_imputer', CryoSleepImputer()),
    ('spending_imputer', SpendingImputer()),
    ('destination_imputer', DestinationImputer()),
    ('VIPImputer', KNNImputerVIP()),
    ('family_feature', FamilyFeatureEngineer()),
    ("Feature_engineering", FeatureEngineer())
])


drop_columns = ['PassengerId', 'Cabin', 'Name', 'Last_Name', 'kfold']  # Example columns to drop

drop_column_transformer = DropUnwantedColumns(columns_to_drop=drop_columns)


def processing(X_train, X_test):
    X_train= name(X_train) #fill names
    X_test= name(X_test) #fill names
    # Fit and transform on training data

    pipeline.fit(X_train)
    X_train = pipeline.transform(X_train)

    # Apply transformations to test data 
    X_test = pipeline.transform(X_test)


    # Drop unwanted columns from both train and test data
    X_train = drop_column_transformer.transform(X_train)
    X_test = drop_column_transformer.transform(X_test)

    Pre=preprocessor
    X_train=Pre.fit_transform(X_train)
    X_test=Pre.transform(X_test)

    return X_train, X_test



