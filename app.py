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
from flask import Flask, abort, request, jsonify, render_template, send_file, session, url_for, redirect, make_response
from flask import Flask, request, jsonify, render_template, send_file, session, url_for, redirect, make_response
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
import itertools

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

WEBHOOK_SECRET = 'qwertyuiopasdfghjklzxcvbnm123456'
DEPLOY_SCRIPT = '/home/ec2-user/x/deploy.sh'


def verify_signature(req):
    signature = req.headers.get('X-Hub-Signature-256')
    if not signature:
        abort(400)
    sha_name, sig = signature.split('=', 1)
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
        if decrypted_response and isinstance(decrypted_response, list) and len(decrypted_response) > 0 and isinstance(
                decrypted_response[0], dict):
            result_row = decrypted_response[0]
        else:
            result_row = decrypted_response

        return result_row or {}

    except Exception as error:
        print(f"🚨 Secure POST Error: {str(error)}")
        raise error


def get_coal_properties_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getCoalPropertiescsv', {"companyId": 1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)
    return csv_output


def get_coal_percentages_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getCoalPercentagescsv', {"companyId": 1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output


def get_Individual_coal_properties_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getIndividualCoalPropertiescsv', {"companyId": 1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output


def get_coke_properties_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getCokePropertiescsv', {"companyId": 1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output


def get_blended_coal_properties_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getBlendedCoalPropertiescsv', {"companyId": 1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output


def get_non_recovery_stamp_charge_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getNonRecoveryStampChargecsv', {"companyId": 1})

    x = response
    y = x[0][0]['csv_output']
    csv_rows = y.split('\n')
    csv_output = '\n'.join(csv_rows)
    print(csv_output)

    return csv_output


def get_coal_count():
    response = post_encrypted('http://3.111.89.109:3000/api/getCoalCount', {"companyId": 1})

    x = response
    y = x[0][0]['csv_output']
    print("y", y)
    return y


def get_min_max_values_csv():
    response = post_encrypted('http://3.111.89.109:3000/api/getMinMaxValuescsv', {"companyId": 1})
    print("minmaxvaluesresponse", response)
    x = response
    y = x[0][0]['csv_output']
    print("minmaxvaluescsv", y)
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
        return jsonify({'status': 'ignored'}), 200

    # Run deploy.sh in the background
    threading.Thread(target=lambda: subprocess.run([DEPLOY_SCRIPT], check=True)).start()
    return jsonify({'status': 'triggered'}), 202


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


# training page
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
            header_format = workbook.add_format(
                {'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'bg_color': '#DDEBF7'})
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
            'charging_tonnage', 'moisture_content', 'bulk_density', 'charging_temperature',
            'battery_operating_temperature', 'cross_wall_temperature', 'push_force',
            'pri', 'coke_per_push', 'gross_coke_yield', 'gcm_pressure', 'gcm_temperature',
            'coking_time', 'coke_end_temperature', 'quenching_time'
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
                coal_type = row.iloc[1]
                current_val = row.iloc[2]
                individual = row.iloc[3:19].tolist()
                blended = row.iloc[19:34].tolist()
                coke_vals = row.iloc[34:41].tolist()
                proc_vals = row.iloc[41:41 + len(process_keys)].tolist()

                # Helper to stringify numeric lists
                def list_to_str(vals):
                    if all(pd.notna(v) and isinstance(v, (int, float)) for v in vals):
                        return '{' + ','.join(str(v) for v in vals) + '}'
                    return ''

                individual_str = list_to_str(individual)
                blended_str = list_to_str(blended)
                coke_str = list_to_str(coke_vals)

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
            'ID', 'Date', 'Coal Type', 'Current Value',
            'Individual Properties', 'Blended Properties',
            'Coke Properties', 'Process Parameters', 'File Name'
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


# FUNCTION TO SAVE THE TRAINING FORM IN CSV

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
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)


@app.route('/submit_training_data', methods=['POST'])
def submit_form():
    data = request.get_json()

    # Save to training data CSV
    save_to_csv(data, coal_data, TRAINING_DATA, )

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
            'sr_no': i + 1,
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
        'ash': {'lower': to_int(row['ash_lower']), 'upper': to_int(row['ash_upper']),
                'default': to_float((row['ash_lower'] + row['ash_upper']) / 2)},
        'vm': {'lower': to_int(row['vm_lower']), 'upper': to_int(row['vm_upper']),
               'default': to_float((row['vm_lower'] + row['vm_upper']) / 2)},
        'm40': {'lower': to_int(row['m40_lower']), 'upper': to_int(row['m40_upper']),
                'default': to_float((row['m40_lower'] + row['m40_upper']) / 2)},
        'm10': {'lower': to_int(row['m10_lower']), 'upper': to_float(row['m10_upper']),
                'default': to_float((row['m10_lower'] + row['m10_upper']) / 2)},
        'csr': {'lower': to_int(row['csr_lower']), 'upper': to_int(row['csr_upper']),
                'default': to_float((row['csr_lower'] + row['csr_upper']) / 2)},
        'cri': {'lower': to_int(row['cri_lower']), 'upper': to_int(row['cri_upper']),
                'default': to_float((row['cri_lower'] + row['cri_upper']) / 2)},
        'ams': {'lower': to_int(row['ams_lower']), 'upper': to_int(row['ams_upper']),
                'default': to_float((row['ams_lower'] + row['ams_upper']) / 2)}
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


# model for cost ai page

GLOBAL_DATA = {}


def initialize_app_startup():
    """Initialize all static data and train models at startup"""
    print("Loading static data and training models at startup...")

    # 1. Load all CSV data (these don't change between requests)
    GLOBAL_DATA['D'] = np.loadtxt(io.StringIO(get_coal_percentages_csv()), delimiter=',')
    GLOBAL_DATA['P'] = np.loadtxt(io.StringIO(get_Individual_coal_properties_csv()), delimiter=',')
    GLOBAL_DATA['Coke_properties'] = np.loadtxt(io.StringIO(get_coke_properties_csv()), delimiter=',')

    # Load coal properties data
    data = pd.read_csv(io.StringIO(get_coal_properties_csv()), dtype=str, header=None, on_bad_lines='skip')
    GLOBAL_DATA['I'] = np.loadtxt(io.StringIO(get_coal_properties_csv()), delimiter=',',
                                  usecols=range(1, data.shape[1] - 2))
    GLOBAL_DATA['coal_data'] = data  # Store for cost calculations

    # Load blended coal parameters
    GLOBAL_DATA['Blended_coal_parameters'] = np.loadtxt(io.StringIO(get_blended_coal_properties_csv()), delimiter=',')

    # Load min/max values
    GLOBAL_DATA['min_max_values'] = read_min_max_values()

    # 2. Load process parameters for different options
    GLOBAL_DATA['Process_parameters'] = {
        1: np.loadtxt('Process_parameter_for_Rec_Top_Char.csv', delimiter=','),
        2: np.loadtxt('Process_parameter_for_Rec_Stam_Char.csv', delimiter=','),
        3: np.loadtxt('Process_parameter_for_Non_Rec_Stam_Char.csv', delimiter=',')
    }

    # 3. Process training data from CSV (this is expensive and static)
    GLOBAL_DATA['training_data'] = process_training_data()

    # 4. Pre-compute daily vectors tensor (expensive computation)
    GLOBAL_DATA['daily_vectors_tensor'] = compute_daily_vectors_tensor(GLOBAL_DATA['D'], GLOBAL_DATA['P'])

    # 5. Train and store both models with scalers
    train_and_store_models()

    print("Startup initialization complete!")


def process_training_data():
    """Process the training CSV data - this is expensive and static"""
    file_path = 'training_data_file.csv'

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
        for row in reader:
            if row[0] not in ('', 'NaT'):
                serial_number = row[0]
                if serial_number not in processed_serial_numbers:
                    coal_percentage = float(row[3])
                    coal_percentages.append(coal_percentage)

                    coal_property_values = [float(val) if val != 'nan' else 0 for val in
                                            row[4].strip('{}').replace(', ', ',').split(',')]
                    coal_properties.append(coal_property_values[:15])

                    if row[6].strip('{}') != '{nan}':
                        coke_output = [float(val) if val != 'nan' else 0 for val in
                                       row[6].strip('{}').replace(', ', ',').split(',')]
                        last_coke_output = coke_output
                    coke_outputs.append(last_coke_output)

                    if row[7].strip('{}') != '{nan}':
                        process_params_str = row[7].replace("'", '"').replace(': ', ':')
                        try:
                            process_params = json.loads(process_params_str)
                            ordered_values = [float(process_params[key]) if key in process_params else 0
                                              for key in process_parameter_keys]
                            last_process_params = ordered_values
                        except json.JSONDecodeError:
                            last_process_params = [0] * len(process_parameter_keys)
                    process_parameters.append(last_process_params)

                    if row[5].strip('{}') != '{nan}':
                        blend_values = [float(val) if val != 'nan' else 0 for val in
                                        row[5].strip('{}').replace(', ', ',').split(',')]
                        last_blend_values = blend_values
                    blends.append(last_blend_values)

                    processed_serial_numbers.add(serial_number)
                else:
                    coal_property_values = [float(val) if val != 'nan' else 0 for val in
                                            row[4].strip('{}').replace(', ', ',').split(',')]
                    coal_properties.append(coal_property_values[:15])

    return {
        'coal_percentages': coal_percentages,
        'coal_properties': coal_properties,
        'blends': blends,
        'Process_parameters': process_parameters,
        'coke_outputs': coke_outputs
    }


def compute_daily_vectors_tensor(D, P):
    """Pre-compute the expensive daily vectors tensor calculation"""
    D_tensor = tf.constant(D, dtype=tf.float32)
    P_tensor = tf.constant(P, dtype=tf.float32)

    daily_vectors = []
    for i in range(D_tensor.shape[0]):
        row_vector = []
        for j in range(P_tensor.shape[1]):
            product_vector = tf.multiply(D_tensor[i], P_tensor[:, j])
            row_vector.append(product_vector)
        daily_vectors.append(tf.stack(row_vector))

    return tf.stack(daily_vectors)


def train_and_store_models():
    """Train both models and store them with their scalers"""
    # Process training data
    training_data = GLOBAL_DATA['training_data']

    # Build blend arrays
    blend_arrays = []
    for i, coal_percentage in enumerate(training_data['coal_percentages']):
        properties_subset = np.array(training_data['coal_properties'][i])
        blend = coal_percentage * properties_subset / 100
        blend_arrays.append(blend)

    blendY = np.array(training_data['blends'])
    blendX = np.array(blend_arrays)

    # Process parameters
    pad_pro_par = [
        np.pad(row, (0, max(0, blendY.shape[1] - len(row))), 'constant') if len(row) < 15 else row
        for row in training_data['Process_parameters']
    ]
    process_par = np.array(pad_pro_par)
    conv_matrix = blendY + process_par

    # Coke output processing
    coke_output = [np.array(row) for row in training_data['coke_outputs']]
    for i in range(len(coke_output)):
        coke_output[i] = np.append(coke_output[i], np.random.uniform(54, 56))

    # Train first model (coal blending model)
    daily_vectors_flattened = GLOBAL_DATA['daily_vectors_tensor'].numpy().reshape(52, -1)

    input_train, input_test, target_train, target_test = train_test_split(
        GLOBAL_DATA['daily_vectors_tensor'].numpy(), GLOBAL_DATA['Blended_coal_parameters'],
        test_size=0.2, random_state=42
    )

    # Scaling for first model
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    input_train_reshaped = input_train.reshape(input_train.shape[0], -1)
    input_test_reshaped = input_test.reshape(input_test.shape[0], -1)

    input_train_scaled = input_scaler.fit_transform(input_train_reshaped)
    input_test_scaled = input_scaler.transform(input_test_reshaped)
    input_train_scaled = input_train_scaled.reshape(-1, 14, 15)
    input_test_scaled = input_test_scaled.reshape(-1, 14, 15)

    target_train_scaled = output_scaler.fit_transform(target_train)
    target_test_scaled = output_scaler.transform(target_test)

    # Build and train first model
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
                   loss='mse', metrics=['mae'])

    modelq.fit(input_train_scaled, target_train_scaled, epochs=100, batch_size=8,
               validation_data=(input_test_scaled, target_test_scaled), verbose=0)

    # Train second model (coke properties model)
    Conv_matrix = GLOBAL_DATA['Blended_coal_parameters'] + GLOBAL_DATA['Process_parameters'][1]  # Default to option 1

    X_train, X_test, y_train, y_test = train_test_split(
        Conv_matrix, GLOBAL_DATA['Coke_properties'], test_size=0.2, random_state=42
    )

    # Scaling for second model
    input__scaler = MinMaxScaler()
    output__scaler = MinMaxScaler()

    input_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    input_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    input_train_scaled = input__scaler.fit_transform(input_train_reshaped)
    input_test_scaled = input__scaler.transform(input_test_reshaped)

    target_train_scaled = output__scaler.fit_transform(y_train)
    target_test_scaled = output__scaler.transform(y_test)

    # Build and train second model
    rf_model = keras.Sequential([
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
                     loss='mse', metrics=['mae'])

    rf_model.fit(input_train_scaled, target_train_scaled, epochs=100, batch_size=8,
                 validation_data=(input_test_scaled, target_test_scaled), verbose=0)

    # Store models and scalers
    GLOBAL_DATA['models'] = {
        'coal_model': modelq,
        'coke_model': rf_model,
        'input_scaler': input_scaler,
        'output_scaler': output_scaler,
        'input_scaler_2': input__scaler,
        'output_scaler_2': output__scaler
    }


def read_min_max_values():
    """Helper function to read min/max values"""
    df = pd.read_csv(io.StringIO(get_min_max_values_csv()))
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

initialize_app_startup()
@app.route('/cost', methods=['POST'])
def cost():
    data = request.json
    if not data:
        raise ValueError("No data received in the request")

    # Extract input data
    coal_blends = data.get("blends")
    coal_types = [blk["coalType"] for blk in coal_blends]
    min_percentages = [int(blk["minPercentage"]) for blk in coal_blends]
    max_percentages = [int(blk["maxPercentage"]) for blk in coal_blends]

    # Pad percentages to 14 elements
    min_percentages_padded = np.pad(min_percentages, (0, 14 - len(min_percentages)), mode='constant')
    max_percentages_padded = np.pad(max_percentages, (0, 14 - len(max_percentages)), mode='constant')

    desired_coke_params = data.get("cokeParameters")
    print("desired coke Parameters:", desired_coke_params)

    # Handle user input blend
    oneblends = data.get('blendcoal', [])
    user_input_values_padded = np.zeros(14)

    if oneblends:
        user_input_values = np.array([blend['currentRange'] for blend in oneblends])
        if user_input_values.sum() != 100:
            return jsonify({"error": "The total of current range must add up to 100."}), 400
        user_input_values_padded = np.pad(user_input_values, (0, 14 - len(user_input_values)), mode='constant')
        user_input_values_padded = np.array(user_input_values_padded).reshape(1, -1)
        print("D-tensor-1", user_input_values_padded)

    print("D-tensor", user_input_values_padded)

    # Process type option
    Option = data.get("processType")
    print(f"Option (type: {type(Option)}):", Option)

    try:
        Option = int(Option)
    except ValueError:
        raise ValueError(f"Invalid option value: {Option} (could not convert to integer)")

    proces_para = data.get("processParameters", {})
    print("model Process Parameters:", proces_para)

    # Get process parameters from global data
    print("Global data",GLOBAL_DATA)
    if Option not in GLOBAL_DATA['Process_parameters']:
        raise ValueError(f"Invalid option value: {Option}")

    Process_parameters = GLOBAL_DATA['Process_parameters'][Option]

    # Pad process parameters if Option == 3
    if Option == 3:
        proces_para = np.pad(proces_para, (0, 2), mode='constant', constant_values=0)

    # Get pre-trained models and scalers
    models = GLOBAL_DATA['models']
    modelq = models['coal_model']
    rf_model = models['coke_model']
    input_scaler = models['input_scaler']
    output_scaler = models['output_scaler']
    input__scaler = models['input_scaler_2']
    output__scaler = models['output_scaler_2']

    # Get other global data
    P = GLOBAL_DATA['P']
    min_max_values = GLOBAL_DATA['min_max_values']

    # Generate all valid combinations
    def generate_combinations(index, current_combination, current_sum):
        target_sum = 100
        if index == len(min_percentages_padded) - 1:
            remaining = target_sum - current_sum
            if min_percentages_padded[index] <= remaining <= max_percentages_padded[index]:
                yield current_combination + [remaining]
            return
        for value in range(min_percentages_padded[index], max_percentages_padded[index] + 1):
            if current_sum + value <= target_sum:
                yield from generate_combinations(index + 1, current_combination + [value], current_sum + value)

    all_combinations = np.array(list(itertools.islice(generate_combinations(0, [], 0), 10000)))

    # Prepare tensors for new combinations
    D_tensor = tf.constant(all_combinations, dtype=tf.float32)
    P_tensor = tf.constant(P, dtype=tf.float32)

    # Compute daily vectors for new combinations
    daily_vectors = []
    for i in range(D_tensor.shape[0]):
        row_vector = []
        for j in range(P_tensor.shape[1]):
            product_vector = tf.multiply(D_tensor[i], P_tensor[:, j])
            row_vector.append(product_vector)
        daily_vectors.append(tf.stack(row_vector))

    daily_vectors_tensor = tf.stack(daily_vectors)
    daily_vectors_flattened = daily_vectors_tensor.numpy().reshape(daily_vectors_tensor.shape[0], -1)

    # Predict blended coal properties using pre-trained model
    b1_scaled = input_scaler.transform(daily_vectors_flattened)
    b1_scaled = b1_scaled.reshape(-1, 14, 15)
    blend1 = modelq.predict(b1_scaled)
    blended_coal_properties = output_scaler.inverse_transform(blend1)

    # Predict coke properties using second pre-trained model
    blend1_with_process = blended_coal_properties + proces_para
    blend1_scaled = input__scaler.transform(blend1_with_process)
    coke = rf_model.predict(blend1_scaled)
    predictions = output__scaler.inverse_transform(coke)

    # Extract desired values
    desired_ash = desired_coke_params["ASH"]
    desired_vm = desired_coke_params["VM"]
    desired_m40 = desired_coke_params["M_40MM"]
    desired_m10 = desired_coke_params["M_10MM"]
    desired_csr = desired_coke_params["CSR"]
    desired_cri = desired_coke_params["CRI"]
    desired_ams = desired_coke_params["AMS"]

    # Filter valid predictions
    def filter_valid_and_invalid(predictions, combinations, blended_coal_properties, min_max_values):
        ash_min = min_max_values['ash']['lower']
        ash_max = min_max_values['ash']['upper']
        vm_min = min_max_values['vm']['lower']
        vm_max = min_max_values['vm']['upper']
        m40_min = min_max_values['m40']['lower']
        m40_max = min_max_values['m40']['upper']
        m10_min = min_max_values['m10']['lower']
        m10_max = min_max_values['m10']['upper']
        csr_min = min_max_values['csr']['lower']
        csr_max = min_max_values['csr']['upper']
        cri_min = min_max_values['cri']['lower']
        cri_max = min_max_values['cri']['upper']
        ams_min = min_max_values['ams']['lower']
        ams_max = min_max_values['ams']['upper']

        print("Min/Max Values:")
        print(f"ASH: {ash_min} to {ash_max}")
        print(f"VM: {vm_min} to {vm_max}")
        print(f"M_40: {m40_min} to {m40_max}")
        print(f"M_10: {m10_min} to {m10_max}")
        print(f"CSR: {csr_min} to {csr_max}")
        print(f"CRI: {cri_min} to {cri_max}")
        print(f"AMS: {ams_min} to {ams_max}")

        valid_indices = []
        invalid_indices = []
        for i, prediction in enumerate(predictions):
            if (ash_min <= prediction[0] <= ash_max and
                    vm_min <= prediction[1] <= vm_max and
                    m40_min <= prediction[9] <= m40_max and
                    m10_min <= prediction[10] <= m10_max and
                    csr_min <= prediction[12] <= csr_max and
                    cri_min <= prediction[13] <= cri_max and
                    ams_min <= prediction[14] <= ams_max):
                valid_indices.append(i)
            else:
                invalid_indices.append(i)

        valid_predictions = predictions[valid_indices]
        valid_combinations = combinations[valid_indices]
        valid_blended_coal_properties = [blended_coal_properties[i] for i in valid_indices]
        invalid_predictions = predictions[invalid_indices]
        invalid_combinations = combinations[invalid_indices]
        invalid_blended_coal_properties = [blended_coal_properties[i] for i in invalid_indices]

        print(f"Number of valid predictions: {len(valid_predictions)}")
        print(f"Number of invalid predictions: {len(invalid_predictions)}")

        return (valid_predictions, valid_combinations, valid_blended_coal_properties,
                invalid_predictions, invalid_combinations, invalid_blended_coal_properties)

    # Filter valid and invalid predictions
    (valid_predictions, valid_combinations, valid_blended_coal_properties,
     invalid_predictions, invalid_combinations, invalid_blended_coal_properties) = \
        filter_valid_and_invalid(predictions, all_combinations, blended_coal_properties, min_max_values)

    predictions = valid_predictions
    all_combinations = valid_combinations
    blended_coal_properties = valid_blended_coal_properties

    # Calculate performance differences
    differences = []
    for prediction in predictions:
        diff = []
        diff.append(((desired_ash - prediction[0]) / desired_ash) * min_max_values['ash']['weight'])
        diff.append(((desired_vm - prediction[1]) / desired_vm) * min_max_values['vm']['weight'])
        diff.append(((prediction[9] - desired_m40) / desired_m40) * min_max_values['m40']['weight'])
        diff.append(((desired_m10 - prediction[10]) / desired_m10) * min_max_values['m10']['weight'])
        diff.append(((prediction[12] - desired_csr) / desired_csr) * min_max_values['csr']['weight'])
        diff.append(((desired_cri - prediction[13]) / desired_cri) * min_max_values['cri']['weight'])
        diff.append(((prediction[14] - desired_ams) / desired_ams) * min_max_values['ams']['weight'])
        differences.append(diff)

    total_differences = [sum(diff) for diff in differences]
    sorted_indices = np.argsort(total_differences)[::-1]

    sorted_predictions = predictions[sorted_indices]
    sorted_blends = all_combinations[sorted_indices]
    sorted_diff = [differences[i] for i in sorted_indices]
    sorted_blended_coal_properties = [blended_coal_properties[i] for i in sorted_indices]

    # Calculate costs
    coal_costs = []
    for i, blend in enumerate(sorted_blends):
        coal_type_costs = []
        for j, coal_type in enumerate(coal_types):
            if j < len(blend):
                coal_type_cost = float(GLOBAL_DATA['coal_data'].loc[GLOBAL_DATA['coal_data'][0] == coal_type,
                GLOBAL_DATA['coal_data'].columns[-1]].values[0])
                coal_type_costs.append(coal_type_cost)
        coal_costs.append(coal_type_costs)

    total_costs = [sum(float(blend[i]) * coal_costs[j][i] / 100
                       for i in range(min(len(blend), len(coal_costs[j]))))
                   for j, blend in enumerate(sorted_blends)]

    print(f"coal_costs: {coal_costs}")

    # Sort by cost
    sorted_indices_by_cost = np.argsort(total_costs)
    sorted_blend_cost = sorted_blends[sorted_indices_by_cost]
    sorted_prediction_cost = sorted_predictions[sorted_indices_by_cost]
    sorted_total_cost = np.array(total_costs)[sorted_indices_by_cost]
    sorted_blended_coal_properties_cost = [sorted_blended_coal_properties[i] for i in sorted_indices_by_cost]
    sorted_diff_cost = [sorted_diff[i] for i in sorted_indices_by_cost]

    # Combine Cost and Performance
    normalized_costs = (total_costs - np.min(total_costs)) / (np.max(total_costs) - np.min(total_costs))
    normalized_differences = ((total_differences - np.min(total_differences)) /
                              (np.max(total_differences) - np.min(total_differences)))

    cost_weight = min_max_values['cost_weightage']
    performance_weight = min_max_values['coke_quality']

    combined_scores = (cost_weight * normalized_costs) + (performance_weight * normalized_differences)
    best_combined_index = np.argmin(combined_scores)

    # Helper for Cost Calculation
    def calculate_cost(blend, coal_costs):
        return sum(blend[i] * coal_costs[0][i] / 100 for i in range(min(len(blend), len(coal_costs[0]))))

    # Pick Three Representative Blends
    # 1. Best by performance (first in performance-sorted list)
    blend_1 = sorted_blends[0]
    blended_coal_1 = sorted_blended_coal_properties[0]
    blend_1_properties = sorted_predictions[0]
    blend_1_cost = calculate_cost(blend_1, coal_costs)

    # 2. Cheapest (first in cost-sorted list)
    blend_2 = sorted_blend_cost[0]
    blended_coal_2 = sorted_blended_coal_properties_cost[0]
    blend_2_properties = sorted_prediction_cost[0]
    blend_2_cost = sorted_total_cost[0]

    # 3. Best combined (from original arrays, using best_combined_index)
    blend_3 = all_combinations[best_combined_index]
    blended_coal_3 = valid_blended_coal_properties[best_combined_index]
    blend_3_properties = valid_predictions[best_combined_index]
    blend_3_cost = calculate_cost(blend_3, coal_costs)

    print(f"Blend 1 Cost: {blend_1_cost}")
    print(f"Blend 2 Cost: {blend_2_cost}")
    print(f"Blend 3 Cost: {blend_3_cost}")

    # Prepare response
    response = {
        "blend1": {
            "composition": blend_1.tolist(),
            "blendedcoal": blended_coal_1.tolist(),
            "properties": blend_1_properties.tolist(),
            "cost": blend_1_cost
        },
        "blend2": {
            "composition": blend_2.tolist(),
            "blendedcoal": blended_coal_2.tolist(),
            "properties": blend_2_properties.tolist(),
            "cost": blend_2_cost
        },
        "blend3": {
            "composition": blend_3.tolist(),
            "blendedcoal": blended_coal_3.tolist(),
            "properties": blend_3_properties.tolist(),
            "cost": blend_3_cost
        },
        "valid_predictions_count": len(valid_predictions)
    }

    # Handle user input prediction if provided
    if np.any(user_input_values_padded != 0):
        D_tensor = tf.constant(user_input_values_padded, dtype=tf.float32)
        print(D_tensor)

        # Compute daily vectors for user input
        daily_vectors = []
        for i in range(D_tensor.shape[0]):
            row_vector = []
            for j in range(P_tensor.shape[1]):
                product_vector = tf.multiply(tf.cast(D_tensor[i], tf.float32), tf.cast(P_tensor[:, j], tf.float32))
                row_vector.append(product_vector)
            daily_vectors.append(tf.stack(row_vector))

        daily_vectors_tensor = tf.stack(daily_vectors)
        daily_vectors_tensor_test_reshaped = daily_vectors_tensor.numpy().reshape(1, -1)
        daily_vectors_tensor_test_scaled = input_scaler.transform(daily_vectors_tensor_test_reshaped)
        daily_vectors_tensor_test_scaled = daily_vectors_tensor_test_scaled.reshape(-1, 14, 15)

        prediction_scaled = modelq.predict(daily_vectors_tensor_test_scaled)
        prediction = output_scaler.inverse_transform(prediction_scaled)

        print("Predicted values:", prediction)

        Conv = proces_para + prediction
        blend1_scaled = input__scaler.transform(Conv)
        coke = rf_model.predict(blend1_scaled)
        predictions_user = output__scaler.inverse_transform(coke)

        print(predictions_user)

        response["ProposedCoal"] = {
            "Blend2": prediction.tolist(),
            "Coke2": predictions_user.tolist()
        }
    else:
        response["ProposedCoal"] = {
            "error": "No valid blendcoal data provided, unable to make predictions."
        }

    return jsonify(response), 200


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

# min-max page


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