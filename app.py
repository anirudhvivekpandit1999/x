import ast
import binascii
import hashlib
import hmac
import subprocess
import threading
import pandas as pd
import numpy as np
import csv
import os
import time
import io, json
from io import BytesIO
from datetime import datetime
from flask import Flask, abort, request, jsonify, render_template, send_file, session,url_for, redirect, make_response
from flask import Flask, request, jsonify, render_template, send_file, session,url_for, redirect, make_response
from flask_cors import CORS
from keras import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from flask_bcrypt import Bcrypt
from google.cloud import translate_v2 as translate
from bs4 import BeautifulSoup
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
import requests
import json
import base64
import binascii
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from itertools import product
import random


app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})

WEBHOOK_SECRET = 'qwertyuiopasdfghjklzxcvbnm123456'
DEPLOY_SCRIPT = '/home/ec2-user/x/deploy.sh'
def verify_signature(req):
        signature = req.headers.get('X-Hub-Signature-256')
        if not signature:
            abort(400)
        sha_name, sig = signature.split('=',1)
        mac = hmac.new(WEBHOOK_SECRET.encode(), req.data, hashlib.sha256)
        if not hmac.compare_digest(mac.hexdigest(), sig):
            abort(400)

def encrypt_data(data):
    """
    Encrypt data using AES-CBC with PKCS7 padding.
    
    Args:
        data: Any JSON-serializable object
        
    Returns:
        Hex string of encrypted data
    """
    # Secret key and IV - same as in JavaScript version
    secret_key = b'qwertyuiopasdfghjklzxcvbnm123456'
    iv = b'1234567890123456'
    
    # Convert data to JSON string and encode as bytes
    json_str = json.dumps(data)
    json_bytes = json_str.encode('utf-8')
    
    # Create cipher object and encrypt
    cipher = AES.new(secret_key, AES.MODE_CBC, iv)
    padded = pad(json_bytes, AES.block_size)
    encrypted_bytes = cipher.encrypt(padded)
    
    # Convert to hex string (equivalent to Base64→Hex in JS version)
    hex_str = binascii.hexlify(encrypted_bytes).decode('utf-8')
    
    return hex_str

def decrypt_data(encrypted_hex_data):
    """
    Decrypt AES-CBC encrypted hex data.
    
    Args:
        encrypted_hex_data: Hex string of encrypted data
        
    Returns:
        Decrypted object or None if decryption fails
    """
    try:
        # Secret key and IV - same as in JavaScript version
        secret_key = b'qwertyuiopasdfghjklzxcvbnm123456'
        iv = b'1234567890123456'
        
        # Input validation
        if not encrypted_hex_data or len(encrypted_hex_data) < 16:
            print("⚠️ Encrypted data is missing or too short:", encrypted_hex_data)
            return None
        
        # Convert hex to bytes
        encrypted_bytes = binascii.unhexlify(encrypted_hex_data)
        
        # Create cipher object and decrypt
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)
        
        # Convert bytes to string and parse JSON
        decrypted_str = decrypted_bytes.decode('utf-8')
        
        try:
            return json.loads(decrypted_str)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON parse failed. Decrypted string: {decrypted_str}")
            raise json_err
            
    except Exception as error:
        print(f"❌ Decryption error: {str(error)}")
        return None

def post_encrypted(endpoint, data):
    """
    Post encrypted data to an endpoint and return decrypted response.
    
    Args:
        endpoint: URL endpoint to post to
        data: Data to encrypt and send
        
    Returns:
        Decrypted response data
    """
    try:
        # Encrypt the payload
        encrypted_payload = encrypt_data(data)
        
        # Send request
        response = requests.post(endpoint, json={"encryptedData": encrypted_payload})
        response_data = response.json()
        
        # Extract encrypted response
        encrypted_response = response_data.get('response')
        
        # Decrypt the response
        decrypted_response = decrypt_data(encrypted_response)
        
        # Process the result similar to JS version
        if decrypted_response and isinstance(decrypted_response, list) and len(decrypted_response) > 0 and isinstance(decrypted_response[0], dict):
            result_row = decrypted_response[0]
        else:
            result_row = decrypted_response
            
        return result_row or {}
        
    except Exception as error:
        print(f"🚨 Secure POST Error: {str(error)}")
        raise error
    
def get_coal_properties_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getCoalPropertiescsv',{"companyId":1})
    
    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)
    return csv_output
def get_coal_percentages_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getCoalPercentagescsv',{"companyId":1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output

def get_Individual_coal_properties_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getIndividualCoalPropertiescsv',{"companyId":1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output

def get_coke_properties_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getCokePropertiescsv',{"companyId":1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output


def get_blended_coal_properties_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getBlendedCoalPropertiescsv',{"companyId":1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output


def get_non_recovery_stamp_charge_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getNonRecoveryStampChargecsv',{"companyId":1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output
def get_coal_count():
    response = post_encrypted('http://3.111.89.109:3000/api/getCoalCount',{"companyId":1})

    x = response
    y = x[0][0]['csv_output']
    print("y",y)
    return y

def get_min_max_values_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getMinMaxValuescsv',{"companyId":1})
    print("minmaxvaluesresponse",response)
    x = response
    y = x[0][0]['csv_output']
    print("minmaxvaluescsv",y)
    return y

coal_count_number = get_coal_count()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/github-deploy', methods=['POST'])
def github_deploy():
        verify_signature(request)
        payload = request.get_json() or {}
        if payload.get('ref') != 'refs/heads/main':
            return jsonify({'status':'ignored'}), 200

        # Run deploy.sh in the background
        threading.Thread(target=lambda: subprocess.run([DEPLOY_SCRIPT], check=True)).start()
        return jsonify({'status':'triggered'}), 202

@app.route('/index.html')
def index_html():
    return render_template('index.html')

@app.route('/hi.html')
def index_html2():
    return render_template('hi.html')

@app.route('/coal-properties.html')
def properties():
    return render_template('coal-properties.html')

@app.route('/min-max.html')
def minmax():
    return render_template('min-max.html')

@app.route('/cost-ai.html')
def costai():
    return render_template('cost-ai.html')

@app.route('/training.html')
def trainig_html():
    return render_template('training.html') 

@app.route('/TrainData-storage.html')
def traindata_html():
    return render_template('TrainData-storage.html') 

@app.route('/login.html')
def login():
    return render_template('login.html')


#training page 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TRAINING_DATA = 'training_data_file.csv'
INDIVIDUAL_UPLOADS_FOLDER = os.path.join(UPLOAD_FOLDER, 'individual_uploads')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(INDIVIDUAL_UPLOADS_FOLDER):
    os.makedirs(INDIVIDUAL_UPLOADS_FOLDER)

def get_next_index():
    # Check if the CSV file exists and is not empty
    if os.path.exists(TRAINING_DATA):
        with open(TRAINING_DATA, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows:
                last_index = 0
                for row in rows:
                    try:
                        

                        if row[0].strip():  
                            last_index = max(last_index, int(row[0]))
                    except ValueError:
                        continue
                return last_index + 1 
            else:
                return 1  # If the CSV is empty, start with 1
    else:
        return 1  # If the CSV doesn't exist, start with 1
    


# UPLOAD EXCEL FILE IN TRAINING PAGE 


@app.route('/download-template', methods=['GET'])
def download_template():
    try:
        # Define the first header (Main Categories)
        main_header = [
            'Date', 'Coal Type', 'Current Percentage',
            'Individual Coal Properties', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
            'Blended Coal Properties', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
            'Coke Properties', '', '', '', '', '', '', '',
            'Process Parameters', '', '', '', '', '', '', '', '', '', '', '', '', '', ''
        ]

        # Define the second header (Subcategories)
        sub_header = [
            '', '', '',
            'Ash', 'VM', 'Moisture', 'Max. Contraction', 'Max. Expansion',
            'Max. fluidity', 'MMR', 'HGI', 'Softening temperature (degC)',
            'Resolidification temp min (degC)', 'Resolidification temp max (degC)',
            'Plastic range (degC)', 'Sulphur', 'Phosphorous', 'CSN', 'Cost (INR)',
            'Ash', 'VM', 'Moisture', 'Max. Contraction', 'Max. Expansion',
            'Max. fluidity', 'Crushing Index <3.15mm', 'Crushing Index <0.5mm',
            'Softening temperature (degC)', 'Resolidification temp min (degC)', 'Resolidification temp max (degC)',
            'Plastic range (degC)', 'Sulphur', 'Phosphorous', 'CSN',
            'Ash', 'VM', 'M40', 'M10', 'CSR', 'CRI', 'AMS',
            'Charging Tonnage', 'Moisture Content', 'Bulk Density',
            'Charging Temperature', 'Battery Operating Temp', 'Cross Wall Temp',
            'Push Force', 'PRI', 'Coke per Push', 'Gross Coke Yield',
            'Gcm Pressure', 'Gcm Temp', 'Coking Time', 'Coke End Temp',
            'Quenching Time', 'Header Temp'
        ]

        # Ensure header count matches
        num_columns = len(sub_header)

        # Example data (3 rows)
        data = [
            ['04-03-2025', 'Coal Type 1', 30] + [''] * (num_columns - 3),
            ['', 'Coal Type 2', 30] + [''] * (num_columns - 3),
            ['', 'Coal Type 3', 40] + [''] * (num_columns - 3),
        ]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Output buffer
        output = BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, startrow=2, header=False, sheet_name='Template')
            workbook = writer.book
            worksheet = writer.sheets['Template']

            # Formats
            header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'bg_color': '#DDEBF7'})
            subheader_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})

            # Merge main headers
            worksheet.merge_range(0, 3, 0, 18, 'Individual Coal Properties', header_format)
            worksheet.merge_range(0, 19, 0, 33, 'Blended Coal Properties', header_format)
            worksheet.merge_range(0, 34, 0, 40, 'Coke Properties', header_format)
            worksheet.merge_range(0, 41, 0, 56, 'Process Parameters', header_format)

            # Static headers
            worksheet.write(0, 0, 'Date', header_format)
            worksheet.write(0, 1, 'Coal Type', header_format)
            worksheet.write(0, 2, 'Current Value', header_format)

            # Sub headers
            for col in range(num_columns):
                worksheet.write(1, col, sub_header[col], subheader_format)

            # Adjust formatting
            worksheet.set_row(0, 25)
            worksheet.set_row(1, 30)
            for col in range(num_columns):
                worksheet.set_column(col, col, max(len(str(sub_header[col])) if sub_header[col] else 15, 15))

        output.seek(0)
        return send_file(output,
                         as_attachment=True,
                         attachment_filename='coal_template.xlsx',
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        print("Error in /download-template:", e)
        return jsonify({'error': str(e)}), 500
    

def format_list_to_string(data_list):
    if not data_list or all(pd.isna(x) for x in data_list):
        return None
    

    formatted_list = []
    for item in data_list:
        if pd.isna(item):
            continue
        if isinstance(item, list):
            formatted_list.append(item)
        elif isinstance(item, (int, float)):
            formatted_list.append(item)
        else:
            try:
                formatted_list.append(float(item))
            except ValueError:
                formatted_list.append(item)
    

    if not formatted_list:
        return None

    if all(isinstance(x, (int, float)) for x in formatted_list):
        return str(formatted_list).replace("'", "")
    else:
        return str([formatted_list]).replace("'", "")

@app.route('/upload_excel_training', methods=['POST'])
def upload_excel_training():
    try:
        # 1. Check file in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # 2. Save it
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 3. Read Excel
        df = pd.read_excel(filepath)

        # 4. Prepare to build new CSV rows
        current_index = get_next_index()
        new_rows = []
        process_keys = [
            'charging_tonnage','moisture_content','bulk_density','charging_temperature',
            'battery_operating_temperature','cross_wall_temperature','push_force',
            'pri','coke_per_push','gross_coke_yield','gcm_pressure','gcm_temperature',
            'coking_time','coke_end_temperature','quenching_time'
        ]

        for i, row in df.iterrows():
            # Detect new date whenever first column is non-null
            if pd.notna(row.iloc[0]):
                try:
                    base_date = pd.to_datetime(row.iloc[0]).date().isoformat()
                except:
                    base_date = str(row.iloc[0])
            # Only process rows with coal type & current value
            if pd.notna(row.iloc[1]) and pd.notna(row.iloc[2]):
                coal_type   = row.iloc[1]
                current_val = row.iloc[2]
                individual  = row.iloc[3:19].tolist()
                blended     = row.iloc[19:34].tolist()
                coke_vals   = row.iloc[34:41].tolist()
                proc_vals   = row.iloc[41:41+len(process_keys)].tolist()

                # Helper to stringify numeric lists
                def list_to_str(vals):
                    if all(pd.notna(v) and isinstance(v, (int, float)) for v in vals):
                        return '{' + ','.join(str(v) for v in vals) + '}'
                    return ''

                individual_str = list_to_str(individual)
                blended_str    = list_to_str(blended)
                coke_str       = list_to_str(coke_vals)

                # Build process dict & stringify if any
                proc_dict = {
                    k: v for k, v in zip(process_keys, proc_vals) if pd.notna(v)
                }
                process_str = str(proc_dict).replace("'", "") if proc_dict else ''

                # Only append if we have the essentials
                if individual_str:
                    new_rows.append([
                        current_index,
                        base_date,
                        coal_type,
                        current_val,
                        individual_str,
                        blended_str,
                        coke_str,
                        process_str,
                        filename
                    ])
                    current_index += 1

        # 5. Append to CSV
        header = [
            'ID','Date','Coal Type','Current Value',
            'Individual Properties','Blended Properties',
            'Coke Properties','Process Parameters','File Name'
        ]
        file_exists = os.path.exists(TRAINING_DATA)
        with open(TRAINING_DATA, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerows(new_rows)

        return jsonify({'message': f'{len(new_rows)} rows imported successfully'}), 200

    except Exception as e:
        # full traceback to console
        app.logger.exception("Error in /upload_excel_training")
        return jsonify({'error': str(e)}), 500
#FUNCTION TO SAVE THE TRAINING FORM IN CSV 

def load_coal_data():
    

    df = pd.read_csv('individual_coal_prop.csv', header=None)
    coal_data = {}
    

    for _, row in df.iterrows():
        coal_type = row[0] 
        properties = row[1:-1].tolist()  
        coal_data[coal_type] = {
            'properties': properties
        }
    

    return coal_data


coal_data = load_coal_data()
       

# Route to fetch coal data for the dropdown (via AJAX)


@app.route('/training_coal_data', methods=['GET'])
def training_coal_data():
    coal_data = load_coal_data()  
    return jsonify(coal_data)

def save_to_csv(data, coal_data, filename):
    index = get_next_index()
    coal_info = data['coalData']
    blended_coal_parameters = data['blendedCoalParameters']
    coke_parameters = data['cokeParameters']
    process_parameters = data['processParameters']

    # Format blended coal parameters
    blended_coal_values = "{" + ",".join([
        str(blended_coal_parameters['ash']),
        str(blended_coal_parameters['volatileMatter']),
        str(blended_coal_parameters['moisture']),
        str(blended_coal_parameters['maxContraction']),
        str(blended_coal_parameters['maxExpansion']),
        str(blended_coal_parameters['maxFluidity']),
        str(blended_coal_parameters['crushingIndex315mm']),
        str(blended_coal_parameters['crushingIndex05mm']),
        str(blended_coal_parameters['softeningTemperature']),
        str(blended_coal_parameters['resolidificationTempRangeMin']),
        str(blended_coal_parameters['resolidificationTempRangeMax']),
        str(blended_coal_parameters['plasticRange']),
        str(blended_coal_parameters['sulphur']),
        str(blended_coal_parameters['phosphorous']),
        str(blended_coal_parameters['csn'])
    ]) + "}"

    # Format coke parameters
    coke_values = "{" + ",".join([
        str(coke_parameters['ash']),
        str(coke_parameters['volatileMatter']),
        str(coke_parameters['m40mm']),
        str(coke_parameters['m10mm']),
        str(coke_parameters['csr']),
        str(coke_parameters['cri']),
        str(coke_parameters['ams'])
    ]) + "}"

    # Format process parameters
    process_values = "{" + ",".join([f"{key}:{value}" for key, value in process_parameters.items()]) + "}"

    # Create rows
    rows = [
        [
            str(index),
            data['date'],
            coal_info[0]['coal'],
            coal_info[0]['currentValue'],
            "{" + ",".join(map(str, coal_data[coal_info[0]['coal']]['properties'])) + "}",
            blended_coal_values,
            coke_values,
            process_values
        ]
    ]

    for i in range(1, len(coal_info)):
        rows.append([
            '',
            '',
            coal_info[i]['coal'],
            coal_info[i]['currentValue'],
            "{" + ",".join(map(str, coal_data[coal_info[i]['coal']]['properties'])) + "}",
            '',
            '',
            ''
        ])

    # Write rows
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile,quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)
 


@app.route('/submit_training_data', methods=['POST'])
def submit_form():
    data = request.get_json()

    # Save to training data CSV
    save_to_csv(data, coal_data, TRAINING_DATA,)

    # Save to individual uploads folder
    individual_filename = os.path.join(INDIVIDUAL_UPLOADS_FOLDER, f"{data['date']}.csv")
    save_to_csv(data, coal_data, individual_filename)

    return jsonify({'message': 'Form submitted successfully'}), 200



# TRAINDATA-STORAGE PAGE 
@app.route('/get_uploaded_files', methods=['GET'])
def get_uploaded_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    file_data = []
    for i, file in enumerate(files):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        timestamp = datetime.fromtimestamp(os.path.getctime(file_path))
        file_data.append({
            'sr_no': i+1,
            'filename': file,
            'upload_date': timestamp.strftime('%d-%m-%Y %H:%M:%S')
        })
    return jsonify(file_data)

@app.route('/delete_uploaded_file', methods=['POST'])
def delete_uploaded_file():
    filename = request.json['filename']
    file_path = os.path.join("uploads", filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        

        # Remove data from training data CSV
            
        df = pd.read_csv(TRAINING_DATA)
        df = df[df['File Name'] != filename]
        df.to_csv(TRAINING_DATA, index=False)

        return jsonify({'message': 'File deleted successfully'}), 200
    return jsonify({'error': 'File not found'}), 404


# cost AI page 
@app.route('/get_coal_types_cost', methods=['GET'])
def get_coal_types():
    # Read the CSV file
    file_path = 'individual_coal_prop.csv' 
    coal_data = pd.read_csv(file_path, header=None)
    

    coal_types = coal_data.iloc[:, 0].tolist() 
    coal_properties = coal_data.iloc[:, :-1].values.tolist() 
    

    return jsonify({
        "coal_types": coal_types,
        "coal_properties": coal_properties
    })
    

@app.route('/get_proposed_coal_types', methods=['GET'])
def get_proposed_coal_types():
    # Replace 'coal_data.csv' with the path to your CSV file
    coal_data = pd.read_csv(io.StringIO(get_coal_properties_csv()), header=None)
    coal_types = coal_data.iloc[:, 0].tolist()
    coal_costs = coal_data.iloc[:, -1].tolist()
    coal_info = [{"type": coal_types[i], "cost": coal_costs[i]} for i in range(len(coal_types))]

    return jsonify({'coal_info': coal_info})
      


def load_csv():
    print("load_csvhit")
    """Load the CSV file and return it as a DataFrame."""
    if os.path.exists(MINMAX_FILE_PATH):
        return pd.read_csv(MINMAX_FILE_PATH)
    else:
        raise FileNotFoundError(f"{CSV_FILE} not found!")

def prepare_ranges():
    """Prepare the range data from the CSV."""

    df = load_csv()
    if df.empty:
        return {}

    # Assuming only one row of data in the CSV
    row = df.iloc[0]

    def to_int(x):
        # If it’s a NumPy scalar, .item() will give you a Python int/float
        return x.item() if hasattr(x, 'item') else int(x)

    def to_float(x):
        return x.item() if hasattr(x, 'item') else float(x)


    ranges = {
        'ash': {'lower': to_int(row['ash_lower']), 'upper': to_int(row['ash_upper']), 'default': to_float((row['ash_lower'] + row['ash_upper']) / 2)},
        'vm': {'lower': to_int(row['vm_lower']), 'upper': to_int(row['vm_upper']), 'default': to_float((row['vm_lower'] + row['vm_upper']) / 2)},
        'm40': {'lower': to_int(row['m40_lower']), 'upper': to_int(row['m40_upper']), 'default': to_float((row['m40_lower'] + row['m40_upper']) / 2)},
        'm10': {'lower': to_int(row['m10_lower']), 'upper': to_float(row['m10_upper']), 'default': to_float((row['m10_lower'] + row['m10_upper']) / 2)},
        'csr': {'lower': to_int(row['csr_lower']), 'upper': to_int(row['csr_upper']), 'default': to_float((row['csr_lower'] + row['csr_upper']) / 2)},
        'cri': {'lower': to_int(row['cri_lower']), 'upper': to_int(row['cri_upper']), 'default':to_float( (row['cri_lower'] + row['cri_upper']) / 2)},
        'ams': {'lower': to_int(row['ams_lower']), 'upper': to_int(row['ams_upper']), 'default': to_float((row['ams_lower'] + row['ams_upper']) / 2)}
    }
    return ranges

@app.route('/get_ranges', methods=['GET'])
def get_ranges():

    try:
        ranges = prepare_ranges()
        return jsonify(ranges)
    except FileNotFoundError as e:
        print(e)
        return jsonify({'error': str(e)}), 404




MAX_COMBINATIONS = 10000


def get_submitted_training_coal_csv(): return open("submitted_training_coal_data.csv").read()


D = np.loadtxt(io.StringIO(get_coal_percentages_csv()), delimiter=",")
P = np.loadtxt(io.StringIO(get_Individual_coal_properties_csv()), delimiter=",")
BLENDED_H = np.loadtxt(io.StringIO(get_blended_coal_properties_csv()), delimiter=",")
COKE_H = np.loadtxt(io.StringIO(get_coke_properties_csv()), delimiter=",")
df_cost = pd.read_csv(io.StringIO(get_coal_properties_csv()))
mm_df = pd.read_csv(io.StringIO(get_min_max_values_csv()))

coal_count = D.shape[1]
features = P.shape[1]
coke_features = COKE_H.shape[1]
stage2_input_d = features + 2

metrics = ['ash', 'vm', 'm40', 'm10', 'csr', 'cri', 'ams']
lower_bounds = mm_df[[m + '_lower' for m in metrics]].iloc[0].values
upper_bounds = mm_df[[m + '_upper' for m in metrics]].iloc[0].values
cost_w = mm_df['cost_weightage'].iloc[0]
quality_w = mm_df['coke_quality'].iloc[0]

def clamp_coke(arr):
    out = arr.copy()
    for i in range(len(metrics)):
        out[:, i] = np.minimum(np.maximum(out[:, i], lower_bounds[i]), upper_bounds[i])
    return out

def clamp_blended_coal_properties(arr):
    bounds = [
        (0.0, 50.0),  # Ash (%)
        (0.0, 100.0),  # VM (%)
        (0.0, 40.0),  # Moisture (%)
        (0, 100),  # Max. Contraction (%)
        (0.0, 100.0),  # Max. Expansion (%)
        (0.0, 200.0),  # Max. fluidity
        (0.0, 100.0),  # Crushing index < 3.15 mm (%)
        (0.0, 100.0),  # Crushing index < 0.5 mm (%)
        (0.0, 500.0),  # Softening temperature (°C)
        (0.0, 500.0),  # Resolidification temp range Min (°C)
        (0.0, 500.0),  # Resolidification temp range Max (°C)
        (0.0, 500.0),  # Plastic range (°C)
        (0.0, 10.0),  # Sulphur (%)
        (0.0, 10.0),  # Phosphorous (%)
        (0.0, 100.0)  # CSN
    ]

    out = arr.copy()
    for i, (min_val, max_val) in enumerate(bounds):
        out[:, i] = np.clip(out[:, i], min_val, max_val)
    return out

def parse_and_append():
    lines = get_submitted_training_coal_csv().strip().splitlines()
    perc, indiv, blend_val, coke_val = [], [], None, None

    for L in lines:
        parts = next(csv.reader([L]))
        if parts[3].strip():
            perc.append(float(parts[3]))
            s = parts[4].replace("{", "[").replace("}", "]")
            indiv.append(ast.literal_eval(s))
        if len(parts) > 5 and parts[5].strip():
            blend_val = np.array(ast.literal_eval(parts[5].replace("{", "[").replace("}", "]")), float)
        if len(parts) > 6 and parts[6].strip():
            coke_val = np.array(ast.literal_eval(parts[6].replace("{", "[").replace("}", "]")), float)

    Dn = np.pad(perc, (0, coal_count - len(perc)), 'constant')[None, :]
    if blend_val is None:
        ip = np.array(indiv)
        blend_val = (ip * np.array(perc)[:, None] / 100).sum(axis=0)
    Bn = blend_val[None, :]
    if coke_val is None:
        raise RuntimeError("No coke output in submitted CSV")
    Cn = np.pad(coke_val, (0, coke_features - len(coke_val)), 'constant')[None, :]
    return Dn, Bn, Cn

D_new, B_new, C_new = parse_and_append()
D = np.vstack([D, D_new])
BLENDED_H = np.vstack([BLENDED_H, B_new])
COKE_H = np.vstack([COKE_H, C_new])

hist_flat = (D[:, :, None] * P[None, :, :] / 100).reshape(D.shape[0], -1)
aug_stage1 = np.hstack([BLENDED_H, np.zeros((BLENDED_H.shape[0], 2))])

input_scaler = MinMaxScaler().fit(hist_flat)
stage1_output_scaler = MinMaxScaler().fit(BLENDED_H)
cost_scaler = MinMaxScaler().fit(aug_stage1)
coke_output_scaler = MinMaxScaler().fit(COKE_H)

modelq = Sequential([
    layers.Input(shape=(coal_count, features)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(features)
])
modelq.compile('adam', 'mse')
X1, y1 = hist_flat.reshape(-1, coal_count, features), BLENDED_H
modelq.fit(X1, y1, epochs=50, batch_size=16, verbose=0)

rf_model = Sequential([
    layers.Input(shape=(stage2_input_d,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(coke_features)
])
rf_model.compile('adam', 'mse')
X2, y2 = np.hstack([stage1_output_scaler.transform(BLENDED_H),
                    np.zeros((BLENDED_H.shape[0], 2))]), COKE_H
rf_model.fit(X2, y2, epochs=50, batch_size=16, verbose=0)

def parse_blends(lst, key):
    a = [int(item[key]) for item in lst]
    return np.pad(a, (0, coal_count - len(a)), 'constant')

def generate_combinations(mins, maxs):
    out = []
    for c in product(*[range(mins[i], maxs[i] + 1) for i in range(coal_count)]):
        if sum(c) == 100:
            out.append(c)
            if len(out) >= 2000:  # max combos limit
                break
    return np.array(out)

@app.route('/cost', methods=['POST'])
def cost():
    data = request.get_json() or {}
    blends = data.get('blends', [])
    oneblend = data.get('blendcoal', [])

    mins = parse_blends(blends, 'minPercentage')
    maxs = parse_blends(blends, 'maxPercentage')

    coal_types_list = [b['coalType'] for b in blends]
    cost_vals = [float(df_cost.columns[-1]) for t in coal_types_list]
    cost_array = np.pad(cost_vals, (0, coal_count - len(cost_vals)), 'constant')

    if oneblend:
        v = np.array([b['currentRange'] for b in oneblend])
        if v.sum() != 100:
            return jsonify(error="Total must be 100"), 400

        bi = (v[:, None] * P) / 100
        flat = bi.reshape(1, -1)
        s1 = modelq.predict(input_scaler.transform(flat).reshape(1, coal_count, features))
        bc = stage1_output_scaler.inverse_transform(s1)[0]

        # Clamp blended coal properties
        bc = clamp_blended_coal_properties(bc[None, :])[0]

        aug = np.hstack([s1, np.zeros((1, 2))])
        cp0 = rf_model.predict(aug)
        cp = coke_output_scaler.inverse_transform(cp0)[0]

        # Strict clamp for coke properties
        column_rules = [
            (0, 14, 17),
            (1, 0.5, 1),
            (9, 90, 93),
            (10, 5, 7),
            (12, 65, 70),
            (13, 22, 26),
            (14, 53, 56)
        ]

        for col, min_val, max_val in column_rules:
            val = cp[col]
            if not (min_val < val < max_val):
                if val < min_val:
                    cp[col] = min_val
                else:
                    cp[col] = max_val

        cost_single = float((v * cost_array).sum() / 100)

        return jsonify(ProposedCoal={
            'BlendedCoal': bc.tolist(),
            'Properties': cp.tolist(),
            'Cost': cost_single
        })

    # MULTI COMBO CASE
    combs = generate_combinations(mins, maxs)
    if combs.size == 0:
        combs = generate_combinations(np.zeros_like(mins), np.full_like(maxs, 100))

    N = len(combs)
    inp3 = (combs[:, :, None] * P[None, :, :]) / 100
    flat = inp3.reshape(N, -1)

    s1_all = modelq.predict(input_scaler.transform(flat).reshape(-1, coal_count, features))
    bc_all = stage1_output_scaler.inverse_transform(s1_all)

    # Clamp blended coal predictions
    bc_all = clamp_blended_coal_properties(bc_all)

    aug_all = np.hstack([s1_all, np.zeros((N, 2))])
    cp0_all = rf_model.predict(aug_all)
    cp_all = coke_output_scaler.inverse_transform(cp0_all)

    # Strict clamp for coke properties in top 3 combos
    column_rules = [
        (0, 14, 17),
        (1, 0.5, 1),
        (9, 90, 93),
        (10, 5, 7),
        (12, 65, 70),
        (13, 22, 26),
        (14, 53, 56)
    ]

    top_idxs = [0, 1, 2]
    for col, min_val, max_val in column_rules:
        for i in top_idxs:
            val = cp_all[i][col]
            if not (min_val < val < max_val):
                if val < min_val:
                    cp_all[i][col] = min_val
                else:
                    cp_all[i][col] = max_val

    costs = (combs * cost_array).sum(axis=1) / 100
    norm_cost = costs / costs.max()
    norm_qual = cp_all.sum(axis=1) / cp_all.sum(axis=1).max()

    perf_idx = np.argsort(-norm_qual)
    cost_idx = np.argsort(costs)
    combo_score = norm_cost * cost_w - norm_qual * quality_w
    comb_idx = np.argsort(combo_score)

    out = {'valid_predictions_count': N,
           'top_combinations': combs[comb_idx][:10].tolist(),
           'top_costs': costs[comb_idx][:10].tolist(),
           'top_coke_properties': cp_all[comb_idx][:10].tolist()}

    return jsonify(out)


CSV_FILE = 'individual_coal_prop.csv'

def read_csv():
    with open(CSV_FILE, mode='r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

# Helper function to write to CSV (overwrites the file)
def write_csv(data):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        

def write1_csv(new_data):
    # Validate that new_data is not None or empty
    if not new_data or not isinstance(new_data, list):
        raise ValueError("Invalid data format. Expected a non-empty list.")

    # Check if the file exists and is not empty
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        with open(CSV_FILE, mode='rb+') as file:
            file.seek(-1, os.SEEK_END)
            last_char = file.read(1)
            # Ensure the file ends with a newline
            if last_char != b'\n':
                file.write(b'\n')

    # Open the file in append mode with newline='' to avoid blank rows
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_data)

@app.route('/get_coal_properties_data', methods=['GET'])
def get_coal_data():
    data = read_csv()   
    if not data:  
        return jsonify({"error": "CSV file is empty or malformed"}), 400
    

    coal_types = [row[0] for row in data if len(row) > 0] 
    if not coal_types:
        return jsonify({"error": "No valid coal types found in the CSV"}), 400

    return jsonify({
        'coal_types': coal_types,
        'coal_data': data
    })
    
@app.route('/download-template-properties', methods=['GET'])
def download_template_prop():
    columns = [
        "Coal", "Type of Coal", "Ash (%)", "Volatile Matter (%)", "Moisture (%)",
        "Max. Contraction", "Max. Expansion", "Max. fluidity (ddpm)", "MMR", "HGI",
        "Softening temperature (°C)", "Resolidification temp range Min (°C)",
        "Resolidification temp range Max (°C)", "Plastic range (°C)",
        "Sulphur (%)", "Phosphorous (%)", "CSN", "Cost per Ton (Rs.)"
    ]

    df = pd.DataFrame(columns=columns)
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(output, as_attachment=True,
                     download_name="coal-properties-template.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route('/add_coal_properties', methods=['POST'])
def add_coal():
    try:
        new_data = request.json.get('data')
        if not new_data:
            return jsonify({'error': 'No data provided'}), 400
        

        new_data.append(datetime.now().strftime('%d %B %Y'))
        

        write1_csv(new_data)
        return jsonify({'message': 'Data added successfully'}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred'}), 500


@app.route('/modify_coal_properties', methods=['POST'])
def modify_coal():
    # Get the data from the request
    request_data = request.get_json()
    coal_index = request_data.get('index')
    modified_data = request_data.get('data')

    # Add the current timestamp for the "Last Modified" column
    timestamp = datetime.now().strftime('%d %B %Y')

    coal_data = read_csv()

    if 0 <= coal_index < len(coal_data):
        modified_data[-1] = timestamp
        coal_data[coal_index] = modified_data
        write_csv(coal_data)

        return jsonify({'message': 'Data updated successfully'}), 200
    else:
        return jsonify({'message': 'Invalid coal index'}), 400

    

#min-max page
    

MINMAX_FILE_PATH = 'min-maxvalues.csv'

@app.route('/minmax_get_data', methods=['GET'])
def get_data():
    if os.path.exists(MINMAX_FILE_PATH):
        df = pd.read_csv(MINMAX_FILE_PATH)
        # Convert the first row to a dictionary
        data = df.iloc[0].to_dict() if not df.empty else {}
        return jsonify(data)
    return jsonify({})  # Return empty data if file doesn't exist

@app.route('/minmax', methods=['POST'])
def min_max():
    # Get the form data
    data = request.get_json()

    # Write data to CSV by overwriting the file
    try:
        with open(MINMAX_FILE_PATH, mode='w', newline='') as file:  # 'w' mode overwrites the file
            writer = csv.DictWriter(file, fieldnames=data.keys())
            # Write header row since we're overwriting the file
            writer.writeheader()
            writer.writerow(data)
        return jsonify({"message": "Data saved successfully!"}), 200
    except Exception as e:
        return jsonify({"message": f"Error saving data: {str(e)}"}), 500





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)