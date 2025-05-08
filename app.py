import ast
import asyncio
from binascii import hexlify, unhexlify
import pandas as pd
import numpy as np
import csv
import os
import time
import io, json
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session,url_for, redirect, make_response
from flask_cors import CORS
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
import MySQLdb.cursors
from google.cloud import translate_v2 as translate
from bs4 import BeautifulSoup
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})


_SECRET_KEY = b"qwertyuiopasdfghjklzxcvbnm123456"
_IV         = b"1234567890123456"

def encrypt_data(payload: dict) -> str:
    plaintext = json.dumps(payload).encode('utf-8')
    padded    = pad(plaintext, AES.block_size, style='pkcs7')
    cipher    = AES.new(_SECRET_KEY, AES.MODE_CBC, iv=_IV)
    ciphertext= cipher.encrypt(padded)
    return hexlify(ciphertext).decode('ascii')

def decrypt_data(encrypted_hex: str) -> dict:
    raw       = unhexlify(encrypted_hex)
    cipher    = AES.new(_SECRET_KEY, AES.MODE_CBC, iv=_IV)
    padded    = cipher.decrypt(raw)
    plaintext = unpad(padded, AES.block_size, style='pkcs7')
    return json.loads(plaintext.decode('utf-8'))

def post_encrypted(url: str, payload: dict, timeout: int = 10) -> dict:
    """
    Mirrors your JS:
     - encryptData
     - axios.post
     - first try result.data.coalProperties
     - fallback to result.data.response
     - decryptData
    """
    encrypted = encrypt_data(payload)

    # synchronous HTTP POST
    resp = requests.post(url, json={"encryptedData": encrypted}, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    print("🔒 Raw server response:", body)

    # exactly like JS: try coalProperties first, then .data.response
    
    enc = body["response"]

    if not enc:
        raise ValueError("No encrypted payload in `coalProperties` or `data.response`")

    # decrypt and return
    return decrypt_data(enc)
def format_number(value):
    """Formats numbers to remove unnecessary decimal places"""
    if isinstance(value, (int, float)):
        return "{0:.5g}".format(value)
    else:
        return str(value)
def getCoalPropertiesCSV():
    response =  post_encrypted('http://3.111.89.109:3000/api/getCoalPropertiesCSV', {"companyId":1}
    );
    rows = response
   

    # 3) Write to an in-memory text buffer
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    # header row
    

    # data rows
    for row in rows:
        writer.writerow(row)
    raw_data = output.getvalue()
    data = [ast.literal_eval(item.replace("'", "\"")) for item in raw_data]
# Define the column order (no headers needed)
    field_order = [
    'CoalName', 'Ash', 'VolatileMatter', 'Moisture', 'MaxContraction',
    'MaxExpansion', 'Maxfluidityddpm', 'MMR', 'HGI', 'SofteningTemperaturec',
    'ResolidificationTempRangeMinc', 'ResolidificationTempRangeMaxc',
    'PlasticRangec', 'Sulphur', 'Phosphorous', 'CSN', 'CostPerTonRs'
]

# Generate CSV output without headers
    output2 = io.StringIO()
    writer2 = csv.writer(
        output,
        quoting=csv.QUOTE_NONE,    # No quotes
        escapechar='\\',           # Escape character if needed
        lineterminator='\n'        # Unix-style line endings
    )

# Write only the data rows
    for entry in data:
        row = [format_number(entry[field]) for field in field_order]
        writer2.writerow(row)

# Get the CSV string
    csv_content = output2.getvalue()
    print(csv_content)
    return output2.getvalue()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def index_html():
    return render_template('index.html')

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
SUBMITTED_CSV_PATH = 'submitted_training_coal_data.csv'
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
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)

        # Remove data from training data CSV
        df = pd.read_csv(TRAINING_DATA)
        df = df[df['File Name'] != filename]
        df.to_csv(TRAINING_DATA, index=False)

        return jsonify({'message': 'File deleted successfully'}), 200
    return jsonify({'error': 'File not found'}), 404

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            df = pd.read_excel(file_path)

            headers_list = ['Ash', 'VM', 'Moisture', 'Max. Contraction', 'Max. Expansion', 'Max. fluidity', 'MMR', 'HGI', 'Softening temperature (degC)',
                            'Resolidification temp min (degC)', 'Resolidification temp max (degC)', 'Plastic range (degC)', 'Sulphur', 'Phosphorous', 'CSN']
            df = df[~df.iloc[:, 3:18].apply(lambda row: all(col in headers_list for col in row.dropna()), axis=1)]

            rows_to_write = []
            index_number = get_next_index()

            for _, row in df.iterrows():
                date = row.get('Date', None)
                coal_type = row.get('Coal Type', None)
                value = row.get('Current Value', None)

                coal_properties = format_list_to_string(row.iloc[3:19].tolist())
                blended_coal_params = format_list_to_string(row.iloc[19:35].tolist())
                coke_params = format_list_to_string(row.iloc[35:42].tolist())
                process_params = format_list_to_string(row.iloc[42:].tolist())

                current_index = index_number if pd.notna(date) else None
                if pd.notna(date):
                    index_number = get_next_index()

                rows_to_write.append([current_index, date, coal_type, value, coal_properties, blended_coal_params, coke_params, process_params])

        except Exception as e:
            return jsonify({'message': 'Error reading the Excel file', 'error': str(e)}), 500

        try:
            with open(SUBMITTED_CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in rows_to_write:
                    writer.writerow(row)
            return jsonify({'message': 'File uploaded and data saved successfully!'}), 200

        except Exception as e:
            return jsonify({'message': 'Error saving data to CSV', 'error': str(e)}), 500

    return jsonify({'message': 'Invalid file type. Only .xls and .xlsx are allowed.'}), 400



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
    coal_data = pd.read_csv('individual_coal_prop.csv', header=None)
    coal_types = coal_data.iloc[:, 0].tolist()
    coal_costs = coal_data.iloc[:, -2].tolist()  
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


#model for cost ai page



#model for cost ai page
def read_min_max_values():
            df = pd.read_csv('min-maxvalues.csv')

            return {
                'ash': {
                    'lower': df['ash_lower'].iloc[0],
                    'upper': df['ash_upper'].iloc[0],
                    'weight': df['ash_weight'].iloc[0]
                },
                'vm': {
                    'lower': df['vm_lower'].iloc[0],
                    'upper': df['vm_upper'].iloc[0],
                    'weight': df['vm_weight'].iloc[0]
                },
                'm40': {
                    'lower': df['m40_lower'].iloc[0],
                    'upper': df['m40_upper'].iloc[0],
                    'weight': df['m40_weight'].iloc[0]
                },
                'm10': {
                    'lower': df['m10_lower'].iloc[0],
                    'upper': df['m10_upper'].iloc[0],
                    'weight': df['m10_weight'].iloc[0]
                },
                'csr': {
                    'lower': df['csr_lower'].iloc[0],
                    'upper': df['csr_upper'].iloc[0],
                    'weight': df['csr_weight'].iloc[0]
                },
                'cri': {
                    'lower': df['cri_lower'].iloc[0],
                    'upper': df['cri_upper'].iloc[0],
                    'weight': df['cri_weight'].iloc[0]
                },
                'ams': {
                    'lower': df['ams_lower'].iloc[0],
                    'upper': df['ams_upper'].iloc[0],
                    'weight': df['ams_weight'].iloc[0]
                },
                'cost_weightage': df['cost_weightage'].iloc[0],
                'coke_quality': df['coke_quality'].iloc[0]
            }

min_max_values = read_min_max_values()


file_path = 'training_data_file.csv'
file_path = 'submitted_training_coal_data.csv'
coal_percentages = []
coal_properties = []
blends = []
process_parameters = []
coke_outputs = []
processed_serial_numbers = set()
process_parameter_keys = [
            'charging_tonnage', 'moisture_content', 'bulk_density', 'charging_temperature', 
            'battery_operating_temperature', 'cross_wall_temperature', 'push_force', 'pri', 
            'coke_per_push', 'gross_coke_yield', 'gcm_pressure', 'gcm_temperature', 
            'coking_time', 'coke_end_temperature', 'quenching_time'
        ]

last_blend_values = None
last_coke_output = None
last_process_params = None

with open(file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            try:
                file_name_index = headers.index('File Name')
            except ValueError : 
                file_name_index = None 
            for row in reader:
                if file_name_index is not None and file_name_index < len(row) :
                    row.pop(file_name_index)
                if row[0] not in ('', 'NaT'):  # Check if the serial number is not empty or NaT
                    serial_number = row[0]
                    if serial_number not in processed_serial_numbers:
                        coal_percentage = float(row[3])
                        coal_percentages.append(coal_percentage)

                        coal_property_values = [float(val) if val != 'nan' else 0 for val in row[4].strip('{}').replace(', ', ',').split(',')]
                        coal_properties.append(coal_property_values[:15])

                        if row[6].strip('{}') != '{nan}':
                            coke_output = [float(val) if val != 'nan' else 0 for val in row[6].strip('{}').replace(', ', ',').split(',')]
                            last_coke_output = coke_output
                        coke_outputs.append(last_coke_output)

                        if row[7].strip('{}') != '{nan}':
                            process_params_str = row[7].replace("'", '"')
                            process_params_str = process_params_str.replace(': ', ':')
                            try:
                                process_params = json.loads(process_params_str)
                                ordered_values = [float(process_params[key]) if key in process_params else 0 for key in process_parameter_keys]
                                last_process_params = ordered_values
                            except json.JSONDecodeError:
                                last_process_params = [0] * len(process_parameter_keys)
                        process_parameters.append(last_process_params)

                        if row[5].strip('{}') != '{nan}':
                            blend_values = [float(val) if val != 'nan' else 0 for val in row[5].strip('{}').replace(', ', ',').split(',')]
                            last_blend_values = blend_values
                        blends.append(last_blend_values)

                        processed_serial_numbers.add(serial_number)
                    else:
                        coal_property_values = [float(val) if val != 'nan' else 0 for val in row[4].strip('{}').replace(', ', ',').split(',')]
                        coal_properties.append(coal_property_values[:15])
blend_arrays = []
for i, coal_percentage in enumerate(coal_percentages):
            properties_subset = np.array(coal_properties[i])
            blend = coal_percentage * properties_subset / 100
            blend_arrays.append(blend)
blendY = np.array(blends)
blendX = np.array(blend_arrays)
pad_pro_par = [
            np.pad(row, (0, max(0, blendY.shape[1] - len(row))), 'constant') if len(row) < 15 else row
            for row in process_parameters
        ]
process_par = np.array(pad_pro_par)
conv_matrix = blendY + process_par
coke_output = [np.array(row) for row in coke_outputs]
for i in range(len(coke_output)):
            coke_output[i] = np.append(coke_output[i], np.random.uniform(54, 56))
D= np.loadtxt('coal_percentages.csv', delimiter=',')  
print(D);



Coke_properties = np.loadtxt('coke_properties.csv', delimiter=',')
data1 = pd.read_csv('individual_coal_prop.csv', dtype=str,header=None, on_bad_lines='skip')       
I = np.loadtxt('individual_coal_prop.csv', delimiter=',', usecols=range(1, data1.shape[1] - 2)) 
D_tensor = tf.constant(D, dtype=tf.float32)
daily_vectors = []

daily_vectors_tensor = tf.stack(daily_vectors)        
input_data = tf.reshape(daily_vectors_tensor, [-1, 14])
daily_vectors_flattened = daily_vectors_tensor.numpy().reshape(52, -1) 
Blended_coal_parameters = np.loadtxt('blended_coal_data.csv', delimiter=',')
      
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()







        # Define model
modelq = keras.Sequential([
layers.Input(shape=(14, 15)),
layers.Flatten(),
layers.BatchNormalization(),
layers.Dense(512, activation='relu'),
layers.Dense(256, activation='leaky_relu', kernel_initializer='he_normal'),
layers.LayerNormalization(),

layers.Dense(256, activation='tanh'),
layers.Dropout(0.3),
layers.Dense(256, activation='leaky_relu', kernel_initializer='he_normal'),
layers.Dropout(0.3),

layers.Dense(128, activation='relu'),
layers.BatchNormalization(),
layers.Dense(128, activation='swish', kernel_initializer='he_normal'),
layers.LayerNormalization(),

layers.Dense(64, activation='relu'),
layers.Dropout(0.2),

layers.Dense(64, activation='swish', kernel_initializer='he_normal'),
layers.Dropout(0.25),

layers.Dense(32, activation='relu'),
layers.BatchNormalization(),

layers.Dense(32, activation='swish', kernel_initializer='he_normal'),
layers.LayerNormalization(),
layers.Dense(15, activation='linear')
        ])

modelq.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae'])
modelq.summary()


        # modelq.fit(input_train_scaled, target_train_scaled, epochs=100, batch_size=8, validation_data=(input_test_scaled, target_test_scaled))

input__scaler = MinMaxScaler()
output__scaler = MinMaxScaler()        
rf_model= keras.Sequential([
            layers.Input(shape=(15, 1)),
            layers.Flatten(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='leaky_relu', kernel_initializer='he_normal'),
            layers.LayerNormalization(),

            layers.Dense(256, activation='tanh'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='leaky_relu', kernel_initializer='he_normal'),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='swish', kernel_initializer='he_normal'),
            layers.LayerNormalization(),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),

            layers.Dense(64, activation='swish', kernel_initializer='he_normal'),
            layers.Dropout(0.25),

            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),

            layers.Dense(32, activation='swish', kernel_initializer='he_normal'),
            layers.LayerNormalization(),
            layers.Dense(15, activation='linear')
        ])

rf_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae'])

daily_vectors = []
differences = []
coal_costs = []

@app.route('/cost', methods=['POST'])
def cost():
    p =  getCoalPropertiesCSV();
    P =  np.loadtxt(p, delimiter=',') 
    print(P);
    P_tensor = tf.constant(P, dtype=tf.float32)
    for i in range(D_tensor.shape[0]):
            row_vector = []
            for j in range(P_tensor.shape[1]):
                product_vector = tf.multiply(D_tensor[i], P_tensor[:, j])
                row_vector.append(product_vector)
            daily_vectors.append(tf.stack(row_vector))
    P_ = P
    input_train, input_test, target_train, target_test = train_test_split(
            daily_vectors_tensor.numpy(), Blended_coal_parameters, test_size=0.2, random_state=42
        ) 
    input_train_reshaped = input_train.reshape(input_train.shape[0], -1)
    input_test_reshaped = input_test.reshape(input_test.shape[0], -1)
    input_train_scaled = input_scaler.fit_transform(input_train_reshaped)
    input_test_scaled = input_scaler.transform(input_test_reshaped)
    input_train_scaled = input_train_scaled.reshape(-1, 14, 15)
    input_test_scaled = input_test_scaled.reshape(-1, 14, 15)
    target_train_scaled = output_scaler.fit_transform(target_train)
    target_test_scaled = output_scaler.transform(target_test)

    input_train_scaled = input_train_scaled.reshape(input_train.shape)
    input_test_scaled = input_test_scaled.reshape(input_test.shape)
    input_train_scaled = input_train_scaled.reshape(-1, 14, 15)
    input_test_scaled = input_test_scaled.reshape(-1, 14, 15)
    y_pred = modelq.predict(input_test_scaled)
    y_pred = output_scaler.inverse_transform(y_pred)
    mse = np.mean((target_test - y_pred) ** 2)
    P_tensor = tf.constant(P_, dtype=tf.float32)






    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received in the request"}), 400
            

        coal_blends = data.get("blends", [])
        if not coal_blends:
            return jsonify({"error": "No blend data provided"}), 400
            

        coal_types = [blk["coalType"] for blk in coal_blends]
        coal_count = len(coal_blends)
        

        min_percentages = np.array([int(blk["minPercentage"]) for blk in coal_blends])
        max_percentages = np.array([int(blk["maxPercentage"]) for blk in coal_blends])
        

        pad_size = 14 - coal_count
        min_percentages_padded = np.pad(min_percentages, (0, pad_size), 'constant')
        max_percentages_padded = np.pad(max_percentages, (0, pad_size), 'constant')
        

        desired_coke_params = data.get("cokeParameters", {})
        if not desired_coke_params:
            return jsonify({"error": "Missing coke parameters"}), 400
        

        oneblends = data.get('blendcoal', [])
        user_input_values_padded = np.zeros(14)
        

        if oneblends:
            user_input_values = np.array([blend['currentRange'] for blend in oneblends])
            if not np.isclose(user_input_values.sum(), 100):
                return jsonify({"error": "The total of current range must add up to 100."}), 400
            

            user_input_values_padded[:len(user_input_values)] = user_input_values
            user_input_values_padded = user_input_values_padded.reshape(1, -1)
        

        try:
            Option = int(data.get("processType", 0))
            if Option not in [1, 2, 3]:
                return jsonify({"error": f"Invalid option value: {Option}"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid process type: {data.get('processType')}"}), 400
        

        proces_para = data.get("processParameters", {})
        

        process_parameter_files = {
            1: 'Process_parameter_for_Rec_Top_Char.csv',
            2: 'Process_parameter_for_Rec_Stam_Char.csv',
            3: 'Process_parameter_for_Non_Rec_Stam_Char.csv'
        }
        

        Process_parameters = np.loadtxt(process_parameter_files[Option], delimiter=',')
        if Option == 3:
            Process_parameters = np.pad(Process_parameters, ((0, 0), (0, 2)), 'constant')
        

        Conv_matrix = Blended_coal_parameters + Process_parameters
        

        X_train, X_test, y_train, y_test = train_test_split(
            Conv_matrix, Coke_properties, test_size=0.2, random_state=42
        )
        

        input_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        input_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        

        input_train_scaled = input__scaler.fit_transform(input_train_reshaped)
        input_test_scaled = input__scaler.transform(input_test_reshaped)
        target_train_scaled = output__scaler.fit_transform(y_train)
        target_test_scaled = output__scaler.transform(y_test)
        

        y_pred = rf_model.predict(input_test_scaled)
        y_pred_inv = output_scaler.inverse_transform(y_pred)
        mse = np.mean(np.square(y_test - y_pred_inv))
        

        def generate_combinations_batch(min_vals, max_vals, coal_count, target_sum=100, batch_size=1000):
            """Generate valid combinations in memory-efficient batches"""
            min_vals = min_vals[:coal_count]
            max_vals = max_vals[:coal_count]
            

            min_required = np.sum(min_vals)
            

            if min_required > target_sum:
                return np.empty((0, 14))  
            

            def generate_batch():
                combinations_batch = []
                

                if coal_count <= 3:
                    if coal_count == 2:
                        v1_values = np.arange(min_vals[0], min(max_vals[0] + 1, target_sum - min_vals[1] + 1))
                        for v1 in v1_values:
                            v2 = target_sum - v1
                            if min_vals[1] <= v2 <= max_vals[1]:
                                combo = np.zeros(14)
                                combo[0] = v1
                                combo[1] = v2
                                combinations_batch.append(combo)
                                if len(combinations_batch) >= batch_size:
                                    yield np.array(combinations_batch)
                                    combinations_batch = []
                    

                    elif coal_count == 3:
                        v1_values = np.arange(min_vals[0], min(max_vals[0] + 1, target_sum - min_vals[1] - min_vals[2] + 1))
                        for v1 in v1_values:
                            v2_values = np.arange(min_vals[1], min(max_vals[1] + 1, target_sum - v1 - min_vals[2] + 1))
                            for v2 in v2_values:
                                v3 = target_sum - v1 - v2
                                if min_vals[2] <= v3 <= max_vals[2]:
                                    combo = np.zeros(14)
                                    combo[0] = v1
                                    combo[1] = v2
                                    combo[2] = v3
                                    combinations_batch.append(combo)
                                    if len(combinations_batch) >= batch_size:
                                        yield np.array(combinations_batch)
                                        combinations_batch = []
                else:
                    current = min_vals.copy()
                    

                    remaining = target_sum - current.sum()
                    

                    for _ in range(batch_size * 5): 
                        current = min_vals.copy()
                        remaining = target_sum - current.sum()
                        

                        for i in range(coal_count - 1):
                            max_additional = min(max_vals[i] - min_vals[i], remaining)
                            if max_additional <= 0:
                                continue
                                

                            additional = np.random.randint(0, max_additional + 1)
                            current[i] += additional
                            remaining -= additional
                        

                        if min_vals[coal_count - 1] <= current[coal_count - 1] + remaining <= max_vals[coal_count - 1]:
                            current[coal_count - 1] += remaining
                            padded = np.zeros(14)
                            padded[:coal_count] = current
                            combinations_batch.append(padded)
                    

                    if combinations_batch:
                        yield np.array(combinations_batch)
                

                if combinations_batch:
                    yield np.array(combinations_batch)
            

            return generate_batch()
        
      


        combination_generator = generate_combinations_batch(
            min_percentages_padded, 
            max_percentages_padded, 
            coal_count,
            batch_size=5000  
        )
        

        best_performance_blend = None
        best_performance_prediction = None
        best_performance_blended_coal = None
        best_performance_cost = float('inf')
        best_performance_score = float('-inf')
        

        cheapest_blend = None
        cheapest_prediction = None
        cheapest_blended_coal = None
        cheapest_cost = float('inf')
        

        best_combined_blend = None
        best_combined_prediction = None
        best_combined_blended_coal = None
        best_combined_cost = float('inf')
        best_combined_score = float('-inf')
        

        coal_cost_map = {row[0]: float(row[-2]) for row in data1.itertuples(index=False)}
        coal_costs = np.array([coal_cost_map.get(coal_type, 0.0) for coal_type in coal_types])
        

        desired_params = {
            "ash": desired_coke_params["ASH"],
            "vm": desired_coke_params["VM"],
            "m40": desired_coke_params["M_40MM"],
            "m10": desired_coke_params["M_10MM"],
            "csr": desired_coke_params["CSR"],
            "cri": desired_coke_params["CRI"],
            "ams": desired_coke_params["AMS"]
        }
        

        weights = np.array([
            min_max_values['ash']['weight'],
            min_max_values['vm']['weight'],
            min_max_values['m40']['weight'],
            min_max_values['m10']['weight'],
            min_max_values['csr']['weight'],
            min_max_values['cri']['weight'],
            min_max_values['ams']['weight']
        ])
        

        target_values = np.array([
            desired_params["ash"],
            desired_params["vm"],
            desired_params["m40"],
            desired_params["m10"],
            desired_params["csr"],
            desired_params["cri"],
            desired_params["ams"]
        ])
        

        lower_better_indices = [0, 1, 3, 5]
        higher_better_indices = [2, 4, 6] 
        

        cost_weight = min_max_values['cost_weightage']
        performance_weight = min_max_values['coke_quality']
        

        total_valid_predictions = 0
        

        for batch_idx, all_combinations in enumerate(combination_generator):
            if len(all_combinations) == 0:
                continue
            

            if Option == 3 and batch_idx == 0:  # Only need to do this once
                proces_para = np.pad(proces_para, (0, 2), 'constant')
            

            D_tensor = tf.constant(all_combinations, dtype=tf.float32)
            

            D_expanded = tf.expand_dims(D_tensor, 2)  # shape: [n, d, 1]
            P_expanded = tf.expand_dims(P_tensor, 0)  # shape: [1, d, m]
            

            daily_vectors = tf.multiply(D_expanded, P_expanded)
            daily_vectors = tf.transpose(daily_vectors, perm=[0, 2, 1])
            

            b1 = daily_vectors.numpy().reshape(daily_vectors.shape[0], -1)
            b1_scaled = input_scaler.transform(b1)
            b1_reshaped = b1.reshape(-1, 14, 15)
            

            prediction_batch_size = min(128, len(b1_reshaped))
            blend1 = modelq.predict(b1_reshaped, batch_size=prediction_batch_size)
            blended_coal_properties = output_scaler.inverse_transform(blend1)
            

            blend1_with_process = blend1 + proces_para
            blend1_flattened = blend1_with_process.reshape(blend1_with_process.shape[0], -1)
            blend1_scaled = input__scaler.transform(blend1_flattened)
            

            coke = rf_model.predict(blend1_scaled, batch_size=prediction_batch_size)
            predictions = output__scaler.inverse_transform(coke)
            

            ash_mask = (min_max_values['ash']['lower'] <= predictions[:, 0]) & (predictions[:, 0] <= min_max_values['ash']['upper'])
            vm_mask = (min_max_values['vm']['lower'] <= predictions[:, 1]) & (predictions[:, 1] <= min_max_values['vm']['upper'])
            m40_mask = (min_max_values['m40']['lower'] <= predictions[:, 9]) & (predictions[:, 9] <= min_max_values['m40']['upper'])
            m10_mask = (min_max_values['m10']['lower'] <= predictions[:, 10]) & (predictions[:, 10] <= min_max_values['m10']['upper'])
            csr_mask = (min_max_values['csr']['lower'] <= predictions[:, 12]) & (predictions[:, 12] <= min_max_values['csr']['upper'])
            cri_mask = (min_max_values['cri']['lower'] <= predictions[:, 13]) & (predictions[:, 13] <= min_max_values['cri']['upper'])
            ams_mask = (min_max_values['ams']['lower'] <= predictions[:, 14]) & (predictions[:, 14] <= min_max_values['ams']['upper'])
            

            valid_mask = ash_mask & vm_mask & m40_mask & m10_mask & csr_mask & cri_mask & ams_mask
            valid_indices = np.where(valid_mask)[0]
            

            total_valid_predictions += len(valid_indices)
            

            if len(valid_indices) == 0:
                continue
            

            valid_predictions = predictions[valid_indices]
            valid_combinations = all_combinations[valid_indices]
            valid_blended_coal_properties = blended_coal_properties[valid_indices]
            

            pred_cols = np.column_stack([
                valid_predictions[:, 0],  
                valid_predictions[:, 1],  
                valid_predictions[:, 9],   
                valid_predictions[:, 10],  
                valid_predictions[:, 12], 
                valid_predictions[:, 13],  
                valid_predictions[:, 14]  
            ])
            

            diff_matrix = np.zeros_like(pred_cols)
            

            diff_matrix[:, lower_better_indices] = (target_values[lower_better_indices] - pred_cols[:, lower_better_indices]) / target_values[lower_better_indices]
            diff_matrix[:, higher_better_indices] = (pred_cols[:, higher_better_indices] - target_values[higher_better_indices]) / target_values[higher_better_indices]
            

            weighted_diffs = diff_matrix * weights
            

            performance_scores = np.sum(weighted_diffs, axis=1)
            

            coal_percentages = valid_combinations[:, :coal_count]
            batch_costs = np.sum(coal_percentages * coal_costs / 100, axis=1)
            

            batch_best_perf_idx = np.argmax(performance_scores)
            batch_best_perf_score = performance_scores[batch_best_perf_idx]
            batch_best_perf_cost = batch_costs[batch_best_perf_idx]
            

            if batch_best_perf_score > best_performance_score:
                best_performance_score = batch_best_perf_score
                best_performance_blend = valid_combinations[batch_best_perf_idx].copy()
                best_performance_prediction = valid_predictions[batch_best_perf_idx].copy()
                best_performance_blended_coal = valid_blended_coal_properties[batch_best_perf_idx].copy()
                best_performance_cost = batch_best_perf_cost
            

            batch_cheapest_idx = np.argmin(batch_costs)
            batch_cheapest_cost = batch_costs[batch_cheapest_idx]
            

            if batch_cheapest_cost < cheapest_cost:
                cheapest_cost = batch_cheapest_cost
                cheapest_blend = valid_combinations[batch_cheapest_idx].copy()
                cheapest_prediction = valid_predictions[batch_cheapest_idx].copy()
                cheapest_blended_coal = valid_blended_coal_properties[batch_cheapest_idx].copy()
            

            if len(batch_costs) > 1 and np.max(batch_costs) > np.min(batch_costs):
                norm_costs = (batch_costs - np.min(batch_costs)) / (np.max(batch_costs) - np.min(batch_costs))
            else:
                norm_costs = np.zeros_like(batch_costs)
                

            if len(performance_scores) > 1 and np.max(performance_scores) > np.min(performance_scores):
                norm_performance = (performance_scores - np.min(performance_scores)) / (np.max(performance_scores) - np.min(performance_scores))
            else:
                norm_performance = np.zeros_like(performance_scores)
            

            combined_scores = (cost_weight * norm_costs) + (performance_weight * norm_performance)
            

            batch_best_combined_idx = np.argmin(combined_scores)
            batch_best_combined_score = combined_scores[batch_best_combined_idx]
            batch_best_combined_cost = batch_costs[batch_best_combined_idx]
            

            if best_combined_blend is None or batch_best_combined_score < best_combined_score:
                best_combined_score = batch_best_combined_score
                best_combined_blend = valid_combinations[batch_best_combined_idx].copy()
                best_combined_prediction = valid_predictions[batch_best_combined_idx].copy()
                best_combined_blended_coal = valid_blended_coal_properties[batch_best_combined_idx].copy()
                best_combined_cost = batch_best_combined_cost
        

        if total_valid_predictions == 0:
            return jsonify({"error": "No valid coal blends found that meet the specified criteria."}), 400
        

        response = {
            "blend1": {
                "composition": best_performance_blend.tolist(),
                "blendedcoal": best_performance_blended_coal.tolist(),
                "properties": best_performance_prediction.tolist(),
                "cost": float(best_performance_cost)
            },
            "blend2": {
                "composition": cheapest_blend.tolist(),
                "blendedcoal": cheapest_blended_coal.tolist(),
                "properties": cheapest_prediction.tolist(),
                "cost": float(cheapest_cost)
            },
            "blend3": {
                "composition": best_combined_blend.tolist(),
                "blendedcoal": best_combined_blended_coal.tolist(),
                "properties": best_combined_prediction.tolist(),
                "cost": float(best_combined_cost)
            },
            "valid_predictions_count": total_valid_predictions
        }
        

        if np.any(user_input_values_padded != 0):
            user_daily_vectors = tf.multiply(
                tf.expand_dims(tf.constant(user_input_values_padded, dtype=tf.float32), 2),
                tf.expand_dims(P_tensor, 0)
            )
            user_daily_vectors = tf.transpose(user_daily_vectors, perm=[0, 2, 1])
            

            user_vectors_reshaped = user_daily_vectors.numpy().reshape(1, -1)
            user_vectors_scaled = input_scaler.transform(user_vectors_reshaped).reshape(-1, 14, 15)
            

            user_prediction_scaled = modelq.predict(user_vectors_scaled)
            user_prediction = output_scaler.inverse_transform(user_prediction_scaled)
            

            user_conv = proces_para + user_prediction
            user_conv_scaled = input__scaler.transform(user_conv)
            user_coke = rf_model.predict(user_conv_scaled)
            user_predictions = output__scaler.inverse_transform(user_coke)
            

            response["ProposedCoal"] = {
                "Blend2": user_prediction.tolist(),
                "Coke2": user_predictions.tolist()
            }
        else:
            response["ProposedCoal"] = {
                "message": "No custom blend data provided."
            }
        

        return jsonify(response), 200
        

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        app.logger.error(f"Error in cost calculation: {error_details}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500 

@app.route('/download-template-properties')
def download_template_properties():
    # Define the column headers for the template
    columns = [
        "Coal", "Ash (%)", "Volatile Matter (%)", "Moisture (%)", "Max. Contraction",
        "Max. Expansion", "Max. fluidity (ddpm)", "MMR", "HGI", "Softening temperature (°C)",
        "Resolidification temp range Min (°C)", "Resolidification temp range Max (°C)",
        "Plastic range (°C)", "Sulphur (%)", "Phosphorous (%)", "CSN", "Cost per Ton (Rs.)"
    ]

    # Create an empty DataFrame with the above columns
    df = pd.DataFrame(columns=columns)

    # Save to a BytesIO object instead of a file on disk
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='CoalTemplate')

    output.seek(0)  # Go to the beginning of the BytesIO stream

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name='coal-properties-template.xlsx',
        as_attachment=True
    )

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

