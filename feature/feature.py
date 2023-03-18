import os

feature_folder = "features"
if not os.path.exists(feature_folder):
    os.makedirs(feature_folder)


class Feature:
    def __init__(self, field_name, json_field_pattern, csv_field_pattern, size):
        self.field_name = field_name
        self.json_field_pattern = json_field_pattern
        self.csv_field_pattern = csv_field_pattern
        self.size = size

    def get_file_path(self):
        return "{}/{}.csv".format(feature_folder, self.field_name)

    def get_json_to_csv_field_mapping(self):
        mapping = {}
        if self.size > 1:
            for i in range(0, self.size):
                json_field = self.json_field_pattern.format(i)
                csv_field = self.csv_field_pattern.format(i)
                mapping[json_field] = csv_field
        else:
            mapping[self.json_field_pattern] = self.csv_field_pattern
        return mapping

    def get_csv_field_names(self):
        if self.size > 1:
            return [self.csv_field_pattern.format(i) for i in range(0, self.size)]
        else:
            return [self.csv_field_pattern]


experience_company_feature = Feature("experience_company", "positions/{}/companyName", "experience/{}/company", 5)
experience_date_feature = Feature("experience_date", "positions/{}/date1", "experience/{}/date_range", 5)
experience_name_feature = Feature("experience_name", "positions/{}/title", "experience/{}/name", 5)
education_feature = Feature("education", "educations/{}/title", "education/{}", 3)
position_feature = Feature("position", "profile/headline", "position", 1)

features = [
    experience_company_feature,
    experience_date_feature,
    experience_name_feature,
    education_feature,
    position_feature
]
