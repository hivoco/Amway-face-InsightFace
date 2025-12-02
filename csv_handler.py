# csv_handler.py
import pandas as pd
import os
import uuid
import boto3
from fastapi import UploadFile

CSV_FOLDER = "CSV"
S3_CSV_IMAGES_PREFIX = "csv_images/"

class CSVHandler:
    def __init__(self, s3_client, bucket_name, region):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.region = region
        
        # Ensure CSV folder exists
        os.makedirs(CSV_FOLDER, exist_ok=True)
    
    def get_csv_path(self, csv_file: str) -> str:
        """Get CSV file path"""
        if csv_file not in ["tailand_data_bangkok", "thailand_data_phuket"]:
            raise ValueError("csv_file must be 'tailand_data_bangkok' or 'thailand_data_phuket'")
        
        csv_filename = "tailand_data_bangkok.csv" if csv_file == "tailand_data_bangkok" else "thailand_data_phuket.csv"
        return os.path.join(CSV_FOLDER, csv_filename)
    
    async def upload_image_to_s3(self, file: UploadFile, ada_no: str) -> str:
        """Upload image to S3 and return URL"""
        try:
            image_bytes = await file.read()
            
            file_ext = os.path.splitext(file.filename)[1] or '.jpg'
            unique_filename = f"{ada_no}_{uuid.uuid4()}{file_ext}"
            s3_key = f"{S3_CSV_IMAGES_PREFIX}{unique_filename}"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_bytes,
                ContentType='image/jpeg'
            )
            
            s3_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            return s3_url
            
        except Exception as e:
            raise Exception(f"S3 upload failed: {e}")
    
    # def update_record(self, csv_file: str, ada_no: str, new_number: str, s3_url: str):
    #     """Update CSV record with new number and photo link"""
    #     csv_path = self.get_csv_path(csv_file)
        
    #     if not os.path.exists(csv_path):
    #         raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    #     # Read CSV
    #     df = pd.read_csv(csv_path)
        
    #     # Check required columns
    #     required_cols = ['ada_no', 'name', 'mobile', 'photo_link']
    #     for col in required_cols:
    #         if col not in df.columns:
    #             raise ValueError(f"CSV missing required column: {col}")
        
    #     # Find matching record
    #     matching_rows = df[df['ada_no'].astype(str) == str(ada_no)]
        
    #     if matching_rows.empty:
    #         return None  # Not found
        
    #     # Add new columns if they don't exist
    #     if 'new_number' not in df.columns:
    #         df['new_number'] = ''
        
    #     if 'new_photo_link' not in df.columns:
    #         df['new_photo_link'] = ''
        
    #     # Update the record
    #     df.loc[df['ada_no'].astype(str) == str(ada_no), 'new_number'] = new_number
    #     df.loc[df['ada_no'].astype(str) == str(ada_no), 'new_photo_link'] = s3_url
        
    #     # Save CSV
    #     df.to_csv(csv_path, index=False)
        
    #     # Return updated record
    #     updated_record = df[df['ada_no'].astype(str) == str(ada_no)].iloc[0].to_dict()
    #     return updated_record

    def update_record(self, csv_file: str, ada_no: str, new_number: str, s3_url: str):
        """Update CSV record with dynamic new columns each time"""
        csv_path = self.get_csv_path(csv_file)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        required_cols = ['ada_no', 'name', 'mobile', 'photo_link']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        # Find matching record
        matching_rows = df[df['ada_no'].astype(str) == str(ada_no)]
        if matching_rows.empty:
            return None  # ADA not found

        # Count how many previous updates exist for this ADA
        count = 0
        for col in df.columns:
            if col.startswith("new_number"):
                count += 1

        # Determine next numbered columns
        if count == 0:
            number_col = "new_number"
            photo_col = "new_photo_link"
        else:
            number_col = f"new_number{count}"
            photo_col = f"new_photo_link{count}"

        # Create columns if not present
        if number_col not in df.columns:
            df[number_col] = ""

        if photo_col not in df.columns:
            df[photo_col] = ""

        # Update the record
        df.loc[df['ada_no'].astype(str) == str(ada_no), number_col] = new_number
        df.loc[df['ada_no'].astype(str) == str(ada_no), photo_col] = s3_url

        # Save CSV
        df.to_csv(csv_path, index=False)

        # Return updated record
        updated = df[df['ada_no'].astype(str) == str(ada_no)].iloc[0].to_dict()
        return updated

    
    def get_record(self, csv_file: str, ada_no: str):
        """Get a specific record by ada_no"""
        csv_path = self.get_csv_path(csv_file)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        matching_rows = df[df['ada_no'].astype(str) == str(ada_no)]
        
        if matching_rows.empty:
            return None
        
        return matching_rows.iloc[0].to_dict()
    
    def list_all_records(self, csv_file: str):
        """List all records from CSV"""
        csv_path = self.get_csv_path(csv_file)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        return df.to_dict('records')