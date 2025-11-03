import json
import os
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm

from general_config import images_path, output_path
from src.utils import exctract_images, unify_string_format


class BaseOCR(ABC):
    def __init__(self) -> None:
        self.data_folder = images_path
        self.output_path = output_path
        self.model_name = None  # Must be set by subclasses


    @abstractmethod
    def run_method(self, image_path):
        """Run inference on one image, and outputs a string corresponding to the text extracted"""
        pass


    def inference_tsv(self, tsv_path, debug_mode=False):
        df = pd.read_csv(tsv_path, delimiter='\t')
        dataset = os.path.basename(tsv_path).split('.')[0]
        images_folder = os.path.join(self.data_folder, dataset+'/')
        output_csv = f"{self.output_path}/{self.model_name}/{dataset}/{self.model_name}_{dataset}.csv"
        if debug_mode:
            output_csv = f"{self.output_path}/{self.model_name}/{dataset}_debug/{self.model_name}_{dataset}.csv"
        if os.path.exists(output_csv):
            print(f"the results of model {self.model_name} on dataset {dataset} is already Done!")
            return output_csv
        if not os.path.exists(images_folder):
            print("Extracting Images!")
            exctract_images(tsv_path, images_folder)
            # exctract_images(tsv_path, images_folder)
        else:
            print("IMAGES FOLDER FOUND!")
        results = []
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
            if index > 5 and debug_mode: # in debug mode, inference only first 5
                break
            image_path = os.path.join(images_folder, str(row['index'])+'.png')
            ocr_res = self.run_method(image_path)
            results.append({
                'index': row['index'],
                'answer': row['answer'],
                'prediction': ocr_res
            })
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"OCR results saved to {output_csv}")
        return output_csv


    def eval_results(self, csv_path: str, dataset: str, debug_mode=False):
        """
        Evaluate OCR results from a CSV with 'answer' and 'prediction' columns.
        Outputs a JSON summary and an extended CSV with correctness flag.
        
        Files are saved in: os.path.join(self.output_path, self.model_name)
        Filenames are based on the input CSV, with '_res' added before '.csv'.

        Args:
            csv_path (str): Path to input CSV with 'answer' and 'prediction' columns.
        """
        if self.model_name is None:
            raise ValueError("model_name must be set before calling eval_results.")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Validate required columns
        if 'answer' not in df.columns or 'prediction' not in df.columns:
            raise ValueError("CSV must contain 'answer' and 'prediction' columns.")

        # Case-insensitive comparison
        df['answer_clean'] = df['answer'].astype(str).str.lower().map(unify_string_format)
        df['pred_clean'] = df['prediction'].astype(str).str.lower().map(unify_string_format)
        df['correct'] = df['answer_clean'] == df['pred_clean']

        # Compute metrics
        total = len(df)
        correct = int(df['correct'].sum())
        ratio = round(correct / total if total > 0 else 0.0, 4)

        # Prepare output directory
        output_dir = os.path.join(self.output_path, self.model_name, dataset)
        if debug_mode:
            output_dir = os.path.join(self.output_path, self.model_name, dataset+"_debug")
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filenames using input CSV name
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        json_output_path = os.path.join(output_dir, f"{base_name}_summary.json")
        csv_output_path = os.path.join(output_dir, f"{base_name}_evaluated.csv")

        # Save JSON result
        results = {
            "total": total,
            "correct": correct,
            "ratio": ratio
        }
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # Save detailed CSV (without helper clean columns)
        save_df = df.drop(columns=['answer_clean', 'pred_clean'])
        save_df.to_csv(csv_output_path, index=False)

        print(f"Evaluation results saved to:\n  {json_output_path}\n  {csv_output_path}")