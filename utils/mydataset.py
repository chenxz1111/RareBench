import json
from datasets import load_dataset


class RareDataset():
    def __init__(self, dataset_name, dataset_path, dataset_type) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        if dataset_path is None:
            if dataset_name in ["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"]:
                self.data = load_dataset('chenxz/RareBench', dataset_name, split='test')
            else:
                raise ERROR("Dataset not found")
        else:
            with open(dataset_path, "r", encoding="utf-8-sig") as f:
                self.data = json.load(f)
        if self.dataset_type == "PHENOTYPE":
            self.patient = self.load_ehr_phenotype_data()

    def load_ehr_phenotype_data(self):
        phenotype_mapping = json.load(open("mapping/phenotype_mapping.json", "r", encoding="utf-8-sig"))
        disease_mapping = json.load(open("mapping/disease_mapping.json", "r", encoding="utf-8-sig"))

        patient = []
       
        for p in self.data:
            if self.dataset_path is None:
                phenotype_list = p['Phenotype']
                disease_list = p['RareDisease']
            else:
                phenotype_list = p[0]
                disease_list = p[1]
            if self.dataset_type == "PHENOTYPE":
                phenotype_list = [phenotype_mapping[phenotype] for phenotype in phenotype_list if phenotype in phenotype_mapping]
                disease_list = [disease_mapping[disease] for disease in disease_list if disease in disease_mapping]
            phenotype = ",".join(phenotype_list)
            disease = ",".join(disease_list)
            patient.append((phenotype, disease))
            
            
        return patient