import json

class RareDataset():
    def __init__(self, dataset_name, dataset_path, dataset_type) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.patient = self.load_data()

    def load_data(self):
        phenotype_mapping = json.load(open("mapping/phenotype_mapping.json", "r", encoding="utf-8-sig"))
        disease_mapping = json.load(open("mapping/disease_mapping.json", "r", encoding="utf-8-sig"))

        patient = []
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        for p in data:
            phenotype_list = p[0]
            disease_list = p[1]
            if self.dataset_type == "PHENOTYPE":
                phenotype_list = [phenotype_mapping[phenotype] for phenotype in phenotype_list if phenotype in phenotype_mapping]
                disease_list = [disease_mapping[disease] for disease in disease_list if disease in disease_mapping]
            phenotype = ",".join(phenotype_list)
            disease = ",".join(disease_list)
            patient.append((phenotype, disease))
            
        return patient