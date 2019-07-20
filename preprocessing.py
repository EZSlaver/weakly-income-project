import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections as col

from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV


def print_df_metadata(df, id_column=None):
    if not id_column:
        id_column = 'new_id_col'
        df = df.copy()
        df['new_id_col'] = range(1, len(df) + 1)

    df.info()
    for col_name in df.columns:
        if col_name in (id_column):
            continue

        n_values = df[col_name].nunique()
        print("Column: {}  |  Type = {}  |  {} Unique Values  ".format(col_name, df[col_name].dtype.name, n_values))
        if is_numeric_column(df, col_name):
            n_nan = np.count_nonzero(np.isnan(df[col_name]))
            n_neg = np.count_nonzero(df[col_name] < 0)
            print('\t Negative Count = {}  |  NaN count = {}'.format(n_neg, n_nan))
        else:
            print("\t" + str(df.groupby(col_name)[id_column].nunique()).replace("\n", "\n\t"))
        # df[col_name].value_counts()


def count_value_pairs_between_columns(df, col1, col2, cond_on_1=None, cond_on_2=None):
    df2 = df[[col1, col2]]

    if cond_on_1:
        df2 = df2[cond_on_1(df[col1])]
    if cond_on_2:
        df2 = df2[cond_on_2(df2[col2])]

    col1, col2 = df[col1], df[col2]
    df2.insert(0, "new_col", list(zip(col1, col2)))
    return df2["new_col"].value_counts().sort_index()


def add_column_by_f_on_columns(df, new_col_name, f, *col_names):
    cols = []
    for name in col_names:
        cols.append(df[name])

    df[new_col_name] = f(*cols)

    # df["experience"] = f(np.maximum(df["age"].values - df["education.num"].values))


def extract_target_column(df, target_name):
    # target = (df["income"] == ">50K") * 1
    # df.drop("income", axis=1, inplace=True)
    target = df[target_name]
    df.drop(target_name, axis=1, inplace=True)
    return target


def normalize_numeric_columns(df):
    for col_name in df.columns:
        if not is_numeric_column(df, col_name):
            col = df[col_name]
            df[col_name] = (col - col.min()) / col.std()


def one_hot(df, drop_origin_columns=True):
    for col_name in df.columns:
        if not is_numeric_column(df, col_name):
            df[col_name] = pd.Categorical(df[col_name])
            dfDummies = pd.get_dummies(df[col_name], prefix=col_name).astype(int, False)
            df = pd.concat([dfDummies, df], axis=1)
            if drop_origin_columns:
                df.drop(col_name, axis=1, inplace=True)

    return df


def get_train_test_random_mask(N, part=.2):
    if isinstance(N, pd.DataFrame):
        N = N.shape[0]

    np.random.seed(999)
    df = pd.DataFrame(np.random.randn(N, 2))
    return np.random.rand(len(df)) < part


def is_numeric_column(df, col: [int, str, pd.DataFrame]):
    if isinstance(col, str):
        col = df[col]
    elif isinstance(col, int):
        col = df.iloc[:, col]

    return np.issubdtype(col.dtype, np.number)


def get_train_test_data(df, target):
    from sklearn.model_selection import train_test_split

    X = df.values
    y = target.values

    X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    return X, X_test, y, y_test


def drop_columns(df, *col_names):
    for name in col_names:
        df.drop(name, axis=1, inplace=True)


def get_train_test_data_split_on_col_value(df, target, col_name, col_value):
    yes_indices = df[col_name] == col_value
    no_indices = df[col_name] != col_value

    set_1 = (df[yes_indices], target[yes_indices])
    set_2 = (df[no_indices], target[no_indices])

    return set_1, set_2


def get_work_experience_column(df, age_col_name, education_num_col_name):
    def internal(row):
        if row[education_num_col_name] <= 8 and row[age_col_name] <= 20:
            return 0
        if row[education_num_col_name] == 9 and row[age_col_name] >= 18:
            return row[age_col_name] - 18
        if row[education_num_col_name] == 10 and row[age_col_name] >= 19:
            return row[age_col_name] - 19
        if row[education_num_col_name] == 11 and row[age_col_name] >= 20:
            return row[age_col_name] - 20
        if row[education_num_col_name] == 12 and row[age_col_name] >= 20:
            return row[age_col_name] - 20
        if row[education_num_col_name] == 13 and row[age_col_name] >= 23:
            return row[age_col_name] - 23
        if row[education_num_col_name] == 14 and row[age_col_name] >= 25:
            return row[age_col_name] - 25
        if row[education_num_col_name] == 15 and row[age_col_name] >= 25:
            return row[age_col_name] - 25
        if row[education_num_col_name] == 16 and row[age_col_name] >= 28:
            return row[age_col_name] - 28
        return 0

    return df.apply(internal, axis=1)


def get_occupation_type_column(df, occupation_col_name):
    def inner(cell):
        if cell in {"Adm-clerical"}:
            return 'Administration'
        if cell in {"Armed-Forces"}:
            return 'Armed-Forces'
        if cell in {"Craft-repair", "Handlers-cleaners", "Machine-op-inspct", "Transport-moving"}:
            return 'Administration'
        if cell in {"Priv-house-serv", "Other-service", "Protective-serv", "Tech-support"}:
            return 'Services'
        if cell in {"Prof-specialty"}:
            return 'Specialist'
        if cell in {"Sales"}:
            return 'Sales'
        if cell in {"Exec-managerial"}:
            return 'Executive'
        if cell in {"Farming-fishing"}:
            return 'Farming'
        return cell

    return df[occupation_col_name].apply(inner)


def get_continents_column(df, country_col_name):
    dict = {
        "Cambodia": "SE-Asia"
        , "Canada": "North-America"
        , "China": "Asia"
        , "Columbia": "South-America"
        , "Cuba": "Latin-America"
        , "Dominican-Republic": "Latin-America"
        , "Ecuador": "South-America"
        , "El-Salvador": "South-America"
        , "England": "West-Europe"
        , "France": "West-Europe"
        , "Germany": "West-Europe"
        , "Greece": "East-Europe"
        , "Guatemala": "Latin-America"
        , "Haiti": "Latin-America"
        , "Holand-Netherlands": "West-Europe"
        , "Honduras": "Latin-America"
        , "Hong": "Asia"
        , "Hungary": "East-Europe"
        , "India": "SE-Asia"
        , "Iran": "Middle-East"
        , "Ireland": "West-Europe"
        , "Italy": "West-Europe"
        , "Jamaica": "Latin-America"
        , "Japan": "Asia_Other"
        , "Laos": "SE-Asia"
        , "Mexico": "Latin-America"
        , "Nicaragua": "Latin-America"
        , "Outlying-US(Guam-USVI-etc)": "Latin-America"
        , "Peru": "South-America"
        , "Philippines": "SE-Asia"
        , "Poland": "East-Europe"
        , "Portugal": "West-Europe"
        , "Puerto-Rico": "Latin-America"
        , "Scotland": "West-Europe"
        , "South": "Africa"
        , "Taiwan": "Asia"
        , "Thailand": "SE-Asia"
        , "Trinadad&Tobago": "Latin-America"
        , "United-States": "North-America"
        , "Vietnam": "SE-Asia"
        , "Yugoslavia": "East-Europe"
        , "?": "Unknown"
    }
    return df[country_col_name].apply(dict.__getitem__)


def get_1994GDP_column(df, country_col_name, gdp_col_name="gdp"):
    GDP_dict = {
        'Afghanistan': -1,
        'Albania': 2668,
        'Algeria': 6779,
        'Angola': 2067,
        'Antigua and Barbuda': 14043,
        'Argentina': 10451,
        'Armenia': 1413,
        'Aruba': 28826,
        'Australia': 22175,
        'Austria': 23698,
        'Azerbaijan': 2971,
        'The Bahamas': 19344,
        'Bahrain': 30737,
        'Bangladesh': 1026,
        'Barbados': 9765,
        'Belarus': 4384,
        'Belgium': 22742,
        'Belize': 4337,
        'Benin': 1069,
        'Bhutan': 1848,
        'Bolivia': 2870,
        'Bosnia and Herzegovina': 2992,
        'Botswana': 5928,
        'Brazil': 8046,
        'Brunei Darussalam': 62558,
        'Bulgaria': 6731,
        'Burkina Faso': 623,
        'Burundi': 588,
        'Cabo Verde': 1978,
        'Cambodia': 759,
        'Cameroon': 1679,
        'Canada': 22773,
        'Central African Republic': 694,
        'Chad': 921,
        'Chile': 8097,
        'China': 1668,
        'Colombia': 5893,
        'Comoros': 1125,
        'Democratic Republic of Congo': 516,
        'Republic of Congo': 3674,
        'Costa Rica': 6421,
        "Côte d'Ivoire": 2054,
        'Croatia': 8650,
        'Cyprus': 18268,
        'Czech Republic': 13938,
        'Denmark': 25356,
        'Djibouti': 2147,
        'Dominica': 4992,
        'Dominican Republic': 4568,
        'Ecuador': 5450,
        'Egypt': 4970,
        'El Salvador': 3721,
        'Equatorial Guinea': 820,
        'Eritrea': 1106,
        'Estonia': 7637,
        'Eswatini': 4326,
        'Ethiopia': 401,
        'Fiji': 4581,
        'Finland': 18921,
        'France': 22244,
        'Gabon': 14209,
        'The Gambia': 1630,
        'Georgia': 1537,
        'Germany': 24224,
        'Ghana': 1494,
        'Greece': 15466,
        'Grenada': 5033,
        'Guatemala': 4013,
        'Guinea': 942,
        'Guinea-Bissau': 1087,
        'Guyana': 2839,
        'Haiti': 1117,
        'Honduras': 2329,
        'Hong Kong': 22652,
        'Hungary': 10750,
        'Iceland': 20706,
        'India': 1420,
        'Indonesia': 4033,
        'Iran': 8874,
        'Iraq': -1,
        'Ireland': 17183,
        'Israel': 15472,
        'Italy': 22870,
        'Jamaica': 6140,
        'Japan': 22700,
        'Jordan': 4846,
        'Kazakhstan': 6473,
        'Kenya': 1704,
        'Kiribati': 1236,
        'South Korea': 10957,
        'Kosovo': -1,
        'Kuwait': 49970,
        'Kyrgyzstan': 1307,
        'Laos': 1498,
        'Latvia': 5795,
        'Lebanon': 7760,
        'Lesotho': 1125,
        'Liberia': -1,
        'Libya': 21327,
        'Lithuania': 6817,
        'Luxembourg': 46312,
        'Macau': -1,
        'North Macedonia': 5820,
        'Madagascar': 1021,
        'Malawi': 472,
        'Malaysia': 9746,
        'Maldives': 4361,
        'Mali': 969,
        'Malta': 15343,
        'Marshall Islands': 1914,
        'Mauritania': 1952,
        'Mauritius': 6346,
        'Mexico': 10145,
        'Federated States of Micronesia': 2078,
        'Moldova': 2219,
        'Mongolia': 3047,
        'Montenegro': -1,
        'Morocco': 3122,
        'Mozambique': 267,
        'Myanmar': 941,
        'Namibia': 4638,
        'Nauru': -1,
        'Nepal': 952,
        'Netherlands': 24683,
        'New Zealand': 17148,
        'Nicaragua': 2037,
        'Niger': 605,
        'Nigeria': 2088,
        'Norway': 34986,
        'Oman': 26726,
        'Pakistan': 2208,
        'Palau': -1,
        'Panama': 6680,
        'Papua New Guinea': 2163,
        'Paraguay': 5973,
        'Peru': 4223,
        'Philippines': 2835,
        'Poland': 7501,
        'Portugal': 15254,
        'Puerto Rico': 18135,
        'Qatar': 54056,
        'Romania': 6926,
        'Russia': 9606,
        'Rwanda': 403,
        'Saint Kitts and Nevis': 11768,
        'Saint Lucia': 7623,
        'Saint Vincent and the Grenadines': 4597,
        'Samoa': 2910,
        'San Marino': -1,
        'São Tomé and Príncipe': 1452,
        'Saudi Arabia': 33326,
        'Senegal': 1550,
        'Serbia': -1,
        'Seychelles': 10493,
        'Sierra Leone': 1071,
        'Singapore': 31040,
        'Slovakia': 8845,
        'Slovenia': 12666,
        'Solomon Islands': 1351,
        'Somalia': -1,
        'South Africa': 6521,
        'South Sudan': -1,
        'Spain': 17519,
        'Sri Lanka': 3054,
        'Sudan': 1787,
        'Suriname': 5696,
        'Sweden': 21546,
        'Switzerland': 32343,
        'Syria': 4024,
        'Taiwan': 14475,
        'Tajikistan': 950,
        'Tanzania': 1016,
        'Thailand': 6298,
        'Timor-Leste': -1,
        'Togo': 972,
        'Tonga': 2833,
        'Trinidad and Tobago': 9282,
        'Tunisia': 4371,
        'Turkey': 7931,
        'Turkmenistan': 2244,
        'Tuvalu': -1,
        'Uganda': 768,
        'Ukraine': 4288,
        'United Arab Emirates': 72744,
        'United Kingdom': 20249,
        'United States': 27674,
        'Uruguay': 8655,
        'Uzbekistan': 1668,
        'Vanuatu': 1698,
        'Venezuela': 11026,
        'Vietnam': 1332,
        'Yemen': 2516,
        'Zambia': 1434,
        'Zimbabwe': 2196,
    }

    def gdp_per_country(country):
        if country in GDP_dict and GDP_dict[country] > 0:
            return GDP_dict[country]
        # probably some 3rd-world country
        return 2000

    return df[country_col_name].apply(gdp_per_country)


def train_xgb_classifier_with_parameters(X, X_test, y, y_test, classifier_parameters=None, fit_parameters=None):
    from xgboost.sklearn import XGBClassifier

    if not classifier_parameters:
        classifier_parameters = {"max_depth": 5,
                                 "min_child_weight": 1,
                                 "learning_rate": 0.1,
                                 "n_estimators": 500,
                                 "n_jobs": -1}

    if not fit_parameters:
        fit_parameters = {}

    clf = XGBClassifier(**classifier_parameters)
    clf.fit(X, y, **fit_parameters)
    return clf.predict(X_test)


def recursive_xgb_param_search(reg, X, y, recursion_iter=2):
    param_grid = {
        'max_depth': np.array(list(range(1, 5))),
        'learning_rate': 10 ** (np.array(range(-10, 0))),
        'subsample': np.array([0.4, 0.6, 0.8, 1.0]),
        'colsample_bytree': np.array([0.4, 0.6, 0.8, 1.]),
        'colsample_bylevel': np.array([0.4, 0.6, 0.8, 1.]),
        'min_child_weight': .01 * np.array(list(range(1, 100, 20)), dtype=float),
        'gamma': np.array(list(range(0, 11, 2)), dtype=float),
        'reg_lambda': [0, .5, 1],
        'n_estimators': 5 * 10 ** (np.array(range(1, 4)))
    }

    param_grid_recursion_op = {
        'max_depth': lambda best: int(np.maximum(1, [best - 1, best, best + 1])),
        'learning_rate': lambda best: best * np.array([.2, .5, 1., 1.5, 2., 4.]),
        'subsample': lambda best: best * np.array([0.1, 0.5, 1.0, 1.5, 2.]),
        'colsample_bytree': lambda best: np.unique(np.minimum(1., best * np.array([0.1, 0.5, 1.0, 1.5, 1.]))),
        'colsample_bylevel': lambda best: np.unique(np.minimum(1., best * np.array([0.1, 0.5, 1.0, 1.5, 1.]))),
        'min_child_weight': lambda best: best * np.array([0.1, 0.5, 1.0, 1.5, 2.]),
        'gamma': lambda best: best * np.array([0.1, 0.5, 1.0, 1.5, 2.]),
        'reg_lambda': lambda best: best * np.array([.2, .5, 1., 1.5, 2., 4.]),
        'n_estimators': lambda best: int(np.maximum(1, (best * np.array([0.1, 0.5, 1.0, 2, 3.])).astype(np.int64)))
    }

    for i in range(recursion_iter):
        rs_regr = RandomizedSearchCV(reg, param_grid, n_iter=100, refit=True, random_state=42, scoring="accuracy")
        rs_regr.fit(X, y)

        param_grid = rs_regr.best_params_.copy()
        for par in param_grid_recursion_op:
            param_grid[par] = param_grid_recursion_op[par](param_grid[par])

    return rs_regr.best_params_

    #
    # for i in range(3):
    #     param_grid = {
    #         'max_depth': grid_reference_values['max_depth'] * .5 np.array(list(range(1, 10))),
    #         'learning_rate': 10. ** (np.array(range(-10, 0))),
    #         'subsample': np.array([0.4, 0.6, 0.8, 1.0]),
    #         'colsample_bytree': np.array([0.4, 0.6, 0.8, 1.0]),
    #         'colsample_bylevel': np.array([0.4, 0.6, 0.8, 1.0]),
    #         'min_child_weight': np.array(list(range(1, 100, 10)), dtype=float),
    #         'gamma': np.array(list(range(0, 11, 2)), dtype=float),
    #         'reg_lambda': 10. ** (np.array(range(-10, 0))),
    #         'n_estimators': np.array(list(range(10, 500, 60)))
    #     }
    #

    # fit_params = {'eval_set': [X,y], "eval_metric": metrics.accuracy_score}
    # fit_params = {}
    #
    # rs_regr = RandomizedSearchCV(reg, param_grid, n_iter=300, refit=True, random_state=42)
    # y_pred = rs_regr.fit(X, y, **fit_params).predict(X_test)
    # metrics.accuracy_score(y_pred, y_test)


def get_stacked_combiner_classifier(X, y, combiner, *stacked_classifiers):
    def stack_transform(X):
        results = pd.DataFrame()
        for i, clf in enumerate(stacked_classifiers):
            result = clf.predict(X)
            results.insert(0, "classifier_" + str(i + 1), result)

        return np.concatenate([X, results.values], axis=1)

    combiner.stack_transform = stack_transform
    combiner.stack_transform_and_fit = lambda X, y: combiner.fit(combiner.stack_transform(X), y)
    combiner.stack_transform_and_predict = lambda X: combiner.predict(combiner.stack_transform(X))

    combiner.stack_transform_and_fit(X, y)

    return combiner


def bin_data_in_column(df, col_name, conditions_dict, new_col_name, remove_col=True, default_value="default"):
    df.insert(df.columns.get_loc(col_name), new_col_name, default_value)
    for data_val in conditions_dict:
        range = conditions_dict[data_val]
        df[new_col_name][(df[col_name] >= range[0]) & (df[col_name] < range[1])] = data_val

    df[new_col_name] = df[new_col_name].astype(np.str)

    if remove_col:
        df.drop(col_name, axis=1, inplace=True)


def get_ratios_of_classes(df, column_name, target_col_name):
    v = df.groupby(column_name)[target_col_name].value_counts().unstack()
    v.fillna(0, inplace=True)
    v.insert(0, "tot", v.iloc[:, 0] + v.iloc[:, 1])
    return v.iloc[:, 2] / v.iloc[:, 0]


class OrderByDisributionTransformer:
    """
    Transforms values in columns to values ordered by their relative probability.
    """

    def __init__(self, target_col, supported_column_values, temp_col_name="temp_col", only_non_numeric_columns=True):
        self.target_col = target_col.copy()
        self.temp_col_name = temp_col_name
        self.only_non_numeric_columns = only_non_numeric_columns
        self._dictionaries = None
        self.supported_column_values = supported_column_values

    def fit(self, df):
        self._dictionaries = {}
        df = df.copy()
        df.insert(df.shape[1], self.temp_col_name, self.target_col)
        for col_name in df.columns:
            if col_name == self.temp_col_name or (self.only_non_numeric_columns and is_numeric_column(df, col_name)):
                continue
            ratios = get_ratios_of_classes(df, col_name, self.temp_col_name)
            if col_name in self.supported_column_values:
                for val in self.supported_column_values[col_name]:
                    if val not in ratios.index:
                        val = pd.Series([0], [val])
                        ratios = ratios.append(val)
            ordered_ratios = ratios.sort_values().index.values
            dict = {}
            for i, val in enumerate(ordered_ratios):
                dict[val] = i
            self._dictionaries[col_name] = dict

    def transform(self, df):
        if not self._dictionaries:
            raise Exception("You must fit before transforming.")

        df_new = df.copy()
        for col in self._dictionaries:
            df_new[col] = df_new[col].apply(self._dictionaries[col].__getitem__)
        return df_new

    def fit_and_transform(self, df):
        self.fit(df)
        return self.transform(df)


def get_all_column_values(df, only_non_numeric_columns=True):
    dict = {}
    for col_name in df.columns:
        if only_non_numeric_columns and is_numeric_column(df, col_name):
            continue
        dict[col_name] = sorted(df[col_name].unique())

    return dict


class PolynomializationTransformer:
    """

    """

    def __init__(self, deg=2, at_position=0):
        self.at_position = at_position
        self.degree = deg
        self._polynomial_parts = None

    def fit(self, df):
        self._polynomial_parts = set()
        for d in range(self.degree):
            poly_parts = self._polynomial_parts.copy()
            for i in range(df.shape[1]):
                if not is_numeric_column(df, i):
                    continue
                if len(self._polynomial_parts) == i:
                    self._polynomial_parts.add((i,))
                else:
                    for poly in poly_parts:
                        self._polynomial_parts.add(tuple(sorted(poly + (i,))))

    def transform(self, df):
        if not self._polynomial_parts:
            raise Exception("You must fit before transforming.")

        df = df.copy()

        col_names = df.columns.values
        for p in self._polynomial_parts:
            p = list(p)
            pname = str(p)
            new_col = np.ones((1, df.shape[0]))
            counter = 2
            while len(p):
                counter -= 1
                new_col *= df[col_names[p.pop()]].values

            if counter <= 0:
                df.insert(self.at_position, pname, new_col[0])

        return df

    def fit_and_transform(self, df):
        self.fit(df)
        return self.transform(df)


class DataSplitClassiffier:

    def __init__(self, class_restricted_to, class_other_than, col_index, split_on_value):
        self.class_restricted_to = class_restricted_to
        self.class_other_than = class_other_than
        self.col_index = col_index
        self.split_on_value = split_on_value

    def fit(self, X, y, fit_params1={}, fit_params2={}):

        X1, X2, y1, y2 = DataSplitClassiffier.split_on_column_value([X, y], self.col_index, self.split_on_value)

        self.class_restricted_to.fit(X1.values, y1.values, **fit_params1)
        self.class_other_than.fit(X2.values, y2.values, **fit_params2)

    @staticmethod
    def split_on_column_value(data, col_index, split_on_value):
        ret = []
        indices = not_indices = None
        for X in data:
            dfX = pd.DataFrame(X)
            if indices is None:
                indices = dfX.iloc[:,col_index] == split_on_value
                not_indices = ~indices
            ret.append(dfX[indices])
            ret.append(dfX[not_indices])

        return tuple(ret)

    def predict(self, X):
        dfX = pd.DataFrame(X)
        dfX.reset_index()

        indices = dfX[self.col_index] == self.split_on_value
        X1 = dfX[indices].values

        indices = ~indices
        X2 = dfX[indices].values

        y1 = self.class_restricted_to.predict(X1)
        y2 = self.class_other_than.predict(X2)

        y = indices.copy()
        y[indices] = y2
        y[~indices] = y1

        return y


class OneHotTransformer:

    def __init__(self, supported_column_values, drop_original_columns=True):
        self.drop_original_columns = drop_original_columns
        self.supported_column_values = supported_column_values
        self._dictionaries = None

    def fit(self, df):
        self._dictionaries = self.supported_column_values
        for col_name in df.columns:
            if not is_numeric_column(df, col_name):
                old = np.empty(0)
                if self._dictionaries[col_name]:
                    old = self._dictionaries[col_name]
                self._dictionaries[col_name] = sorted(np.union1d(df[col_name].unique(), old))

    def transform(self, df):
        if not self._dictionaries:
            raise Exception("You must fit before transforming.")

        df = df.copy()
        for col_name in df.columns:
            if col_name not in self._dictionaries:
                continue
            pos = df.columns.get_loc(col_name)
            dfDummies = pd.get_dummies(df[col_name], prefix=col_name).astype(int, False)
            dfDummies = dfDummies.reindex(sorted(dfDummies.columns), axis=1)
            # check column order and that nothing is missing
            i = 0
            for val in self._dictionaries[col_name]:
                col_formatted_name = "{}_{}".format(col_name, val)
                if dfDummies.columns[i] != col_formatted_name:
                    dfDummies.insert(0, col_formatted_name, np.zeros((dfDummies.shape[0], 1)))
                i += 1

            for i, c_name in enumerate(dfDummies.columns):
                df.insert(pos + i, c_name, dfDummies[c_name])

            if self.drop_original_columns:
                df.drop(col_name, axis=1, inplace=True)
            #
            # col_df = pd.DataFrame(np.zeros((df.shape[0],len(self._dictionaries[col_name]))))
            # index_col = df[col_name].apply(lambda x: np.where(self._dictionaries[col_name] == x))
            #
            # df[col_name] = pd.Categorical(df[col_name])
            # dfDummies = pd.get_dummies(df[col_name], prefix=col_name).astype(int, False)
            # df = pd.concat([dfDummies, df], axis=1)
            # if drop_origin_columns:
            #     df.drop(col_name, axis=1, inplace=True)

        return df

    def fit_and_transform(self, df):
        self.fit(df)
        return self.transform(df)


# def one_hot(df, drop_origin_columns=True):
#     for col_name in df.columns:
#         if not is_numeric_column(df, col_name):
#             df[col_name] = pd.Categorical(df[col_name])
#             dfDummies = pd.get_dummies(df[col_name], prefix=col_name).astype(int, False)
#             df = pd.concat([dfDummies, df], axis=1)
#             if drop_origin_columns:
#                 df.drop(col_name, axis=1, inplace=True)
#
#     return df

#
#
# def get_ratio_data_transformation(df, ):
#     dictionaries = {}
#     df = df.copy()
#     df.insert(df.shape[1], temp_col_name, target_col)
#     for col_name in df.columns:
#         if col_name == temp_col_name or (only_non_numeric_columns and is_numeric_column(df, col_name)):
#             continue
#         ratios = get_ratios_of_classes(df, col_name, temp_col_name)
#         ordered_ratios = ratios.sort_values().index.values
#         dict = {}
#         for i, val in enumerate(ordered_ratios):
#             dict[val] = i
#         dictionaries[col_name] = dict
#
#     def transform(df_new):
#         df_new = df_new.copy()
#         for col in dictionaries:
#             df_new[col] = df_new[col].apply(dictionaries[col].__getitem__)
#         return df_new
#
#     return transform


if __name__ == "__main__":
    df_full = pd.read_csv('yearly_income_train.csv')
    df = df_full

    df["1994_gdp"] = get_1994GDP_column(df, "native.country")
    df["country_continent"] = get_continents_column(df, "native.country")
    df["occupation_type"] = get_occupation_type_column(df, "occupation")
    df["experience_years"] = get_work_experience_column(df, "age", "education.num")

    bin_data_in_column(df, "hours.per.week", {
        "hardly": (0, 10),
        "part-time": (10, 30),
        "full-time": (30, 45),
        "over-time": (45, 65),
        "work-on-the-roads": (65, 80),
        "never-stop-working": (80, 100)
    }, "work_dur_def", remove_col=False)

    bin_data_in_column(df, "age", {
        "no-children": (0, 25),
        "1-child": (25, 35),
        "2-children": (35, 100)
    }, "number_of_children", remove_col=False)

    df["capital_net"] = df["capital.gain"] - df["capital.loss"]

    drop_columns(df, "Unnamed: 0", "fnlwgt", "capital.gain", "capital.loss")

    target = (extract_target_column(df, "income") == ">50K") * 1

    dividing_mask = get_train_test_random_mask(df.shape[0])

    df_2 = df.copy()

    col_vals = get_all_column_values(df_2)
    col_vals["native.country"] = sorted(np.union1d(col_vals["native.country"], ["Holand-Netherlands"]))

    df_2_train, target_train = df_2[~dividing_mask], target[~dividing_mask]
    df_2_test, target_test = df_2[dividing_mask], target[dividing_mask]

    order_trf = OrderByDisributionTransformer(target_train, col_vals)

    df_2_train = order_trf.fit_and_transform(df_2_train)
    df_2_test = order_trf.transform(df_2_test)

    poly = PolynomializationTransformer(2)
    df_2_train = poly.fit_and_transform(df_2_train)
    df_2_test = poly.transform(df_2_test)

    # pca = PCA(n_components=50, random_state=82)
    #
    # y, y_test = target_train.values, target_test.values
    # X = pca.fit_transform(df_2_train.values, y)
    # X_test = pca.transform(df_2_test.values)

    X, y = df_2_train.values, target_train.values
    X_test, y_test = df_2_test.values, target_test.values

    from xgboost.sklearn import XGBClassifier
    import sklearn.metrics as metrics

    classifier_parameters = {'subsample': 0.6,
                             'reg_lambda': 1,
                             'n_estimators': 1000,
                             'min_child_weight': 0.81,
                             'max_depth': 2,
                             'learning_rate': 0.05,
                             # 'eval_set': [(X, y), (X_test, y_test)],
                             'gamma': 6.0}

    clf1 = XGBClassifier(**classifier_parameters)
    clf2 = XGBClassifier(**classifier_parameters)
    # clf.fit(X, y)

    col_pos = df_2_train.columns.get_loc("marital.status")
    clf = DataSplitClassiffier(clf1, clf2, col_pos, 5)
    X1, X2, y1, y2 \
        = DataSplitClassiffier.split_on_column_value([X, y],col_pos, 5)
    X_test1, X_test2, y_test1, y_test2 \
        = DataSplitClassiffier.split_on_column_value([X_test, y_test], col_pos, 5)

    clf.fit(X, y,
            {"eval_set": [(X1.values, y1.values), (X_test1.values, y_test1.values)]},
            {"eval_set": [(X2.values, y2.values), (X_test2.values, y_test2.values)]})

    y_pred = clf.predict(X_test)
    ret = metrics.accuracy_score(y_pred, y_test)
    print(ret)

    df_test = pd.read_csv('yearly_income_test_samples.csv')
    df_test["1994_gdp"] = get_1994GDP_column(df_test, "native.country")
    df_test["country_continent"] = get_continents_column(df_test, "native.country")
    df_test["occupation_type"] = get_occupation_type_column(df_test, "occupation")
    df_test["experience_years"] = get_work_experience_column(df_test, "age", "education.num")

    bin_data_in_column(df_test, "hours.per.week", {
        "hardly": (0, 10),
        "part-time": (10, 30),
        "full-time": (30, 45),
        "over-time": (45, 65),
        "work-on-the-roads": (65, 80),
        "never-stop-working": (80, 100)
    }, "work_dur_def", remove_col=False)

    bin_data_in_column(df_test, "age", {
        "no-children": (0, 25),
        "1-child": (25, 35),
        "2-children": (35, 100)
    }, "number_of_children", remove_col=False)

    df_test["capital_net"] = df_test["capital.gain"] - df_test["capital.loss"]
    drop_columns(df_test, "Unnamed: 0", "fnlwgt", "capital.gain", "capital.loss")
    df_test = order_trf.transform(df_test)
    df_test = poly.transform(df_test)

    y_pred = clf.predict(df_test.values)

    out = np.array([">50K" if i == 1 else "<=50K" for i in y_pred])
    pd.DataFrame(out).to_csv("results.csv", header=False, index=False)
