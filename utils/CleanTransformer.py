from .preprocessing import parse_release_date, parse_ts_listen

class CleanTransformer():
    """
    This class allows you to transform the date columns of your dataframe
    and clean their values (taking out the outliers and strange values)
    """
    def __init__(self):
        self.data = None
        self.columns = None
        self.fitted = False
        self.transformed = False

    def fit(self, data):
        self.fitted = True
        self.data = data
        self.columns = data.columns

    def transform(self):
        self.transformed = True
        parse_ts_listen(self.data, drop_tmp=True)
        parse_release_date(self.data)

    def fit_transform(self, data):
        self.fit(data)
        self.transform()

    # TODO: add other features to clean if needed
    def clean_ts_listen(self):
        self.data = self.data[2011 <= self.data["year_listen"] <= 2017]
        self.data = self.data[1 <= self.data["month_listen"] <= 12]
        self.data = self.data[1 <= self.data["day_listen"] <= 31]
        self.data = self.data[0 <= self.data["hour_listen"] <= 23]

    def clean_release_date(self):
        # TODO: verify min and max year release allowed
        self.data = self.data[2011 <= self.data["year_listen"] <= 2017]
        self.data = self.data[1 <= self.data["month_release"] <= 12]
        self.data = self.data[1 <= self.data["day_release"] <= 31]

    def remove_outliers(self):
        # TODO: to implement
        pass

    def clean(self):
        if not self.transformed:
            raise IOError("You need first to transform your dataframe.")
        self.clean_ts_listen()
        self.clean_release_date()