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


app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})




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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_next_index():
    # Check if the CSV file exists and is not empty
    if os.path.exists(SUBMITTED_CSV_PATH):
        with open(SUBMITTED_CSV_PATH, 'r') as f:
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
    

@app.route('/download-template', methods=['GET'])
def download_template():
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

    # Example data
    data = [
        ['04-03-2025', 'Coal Type 1', 30] + [''] * (len(sub_header) - 3),
        ['', 'Coal Type 2', 30] + [''] * (len(sub_header) - 3),
        ['', 'Coal Type 3', 40] + [''] * (len(sub_header) - 3),
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to Excel with XlsxWriter
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, startrow= 2, header=False, sheet_name='Template')
        workbook = writer.book
        worksheet = writer.sheets['Template']

        # Apply header formatting
        header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'bg_color': '#DDEBF7'})
        subheader_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
        
        # Merge the main headers with proper ranges
        worksheet.merge_range(0, 3, 0, 18, 'Individual Coal Properties', header_format)
        worksheet.merge_range(0, 19, 0, 33, 'Blended Coal Properties', header_format)
        worksheet.merge_range(0, 34, 0, 40, 'Coke Properties', header_format)
        worksheet.merge_range(0, 41, 0, 56, 'Process Parameters', header_format)

        # Write the static columns for main header
        worksheet.write(0, 0, 'Date', header_format)
        worksheet.write(0, 1, 'Coal Type', header_format)
        worksheet.write(0, 2, 'Current Value', header_format)

        # Write sub-headers
        for col in range(len(sub_header)):
            worksheet.write(1, col, sub_header[col], subheader_format)

        # Adjust row height for better visibility
        worksheet.set_row(0, 25)
        worksheet.set_row(1, 30)

        # Set column widths
        for col in range(len(sub_header)):
            worksheet.set_column(col, col, max(len(str(sub_header[col])) if sub_header[col] else 15, 15))

    output.seek(0)
    return send_file(output, as_attachment=True, download_name='coal_template.xlsx',
                      mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

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
    ranges = {
        'ash': {'lower': row['ash_lower'], 'upper': row['ash_upper'], 'default': (row['ash_lower'] + row['ash_upper']) / 2},
        'vm': {'lower': row['vm_lower'], 'upper': row['vm_upper'], 'default': (row['vm_lower'] + row['vm_upper']) / 2},
        'm40': {'lower': row['m40_lower'], 'upper': row['m40_upper'], 'default': (row['m40_lower'] + row['m40_upper']) / 2},
        'm10': {'lower': row['m10_lower'], 'upper': row['m10_upper'], 'default': (row['m10_lower'] + row['m10_upper']) / 2},
        'csr': {'lower': row['csr_lower'], 'upper': row['csr_upper'], 'default': (row['csr_lower'] + row['csr_upper']) / 2},
        'cri': {'lower': row['cri_lower'], 'upper': row['cri_upper'], 'default': (row['cri_lower'] + row['cri_upper']) / 2},
        'ams': {'lower': row['ams_lower'], 'upper': row['ams_upper'], 'default': (row['ams_lower'] + row['ams_upper']) / 2}
    }
    return ranges

@app.route('/get_ranges', methods=['GET'])
def get_ranges():
    """Endpoint to fetch slider ranges."""
    try:
        ranges = prepare_ranges()
        return jsonify(ranges)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
        

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
            for row in reader:
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
P =  np.loadtxt('Individual_coal_properties.csv', delimiter=',')  
Coke_properties = np.loadtxt('coke_properties.csv', delimiter=',')
data1 = pd.read_csv('individual_coal_prop.csv', dtype=str,header=None, on_bad_lines='skip')       
I = np.loadtxt('individual_coal_prop.csv', delimiter=',', usecols=range(1, data1.shape[1] - 2)) 
D_tensor = tf.constant(D, dtype=tf.float32)
P_tensor = tf.constant(P, dtype=tf.float32)
daily_vectors = []
for i in range(D_tensor.shape[0]):
            row_vector = []
            for j in range(P_tensor.shape[1]):
                product_vector = tf.multiply(D_tensor[i], P_tensor[:, j])
                row_vector.append(product_vector)
            daily_vectors.append(tf.stack(row_vector))
daily_vectors_tensor = tf.stack(daily_vectors)        
input_data = tf.reshape(daily_vectors_tensor, [-1, 14])
daily_vectors_flattened = daily_vectors_tensor.numpy().reshape(52, -1) 
Blended_coal_parameters = np.loadtxt('blended_coal_data.csv', delimiter=',')
input_train, input_test, target_train, target_test = train_test_split(
            daily_vectors_tensor.numpy(), Blended_coal_parameters, test_size=0.2, random_state=42
        )       
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
        
input_train_scaled = input_train_scaled.reshape(input_train.shape)
input_test_scaled = input_test_scaled.reshape(input_test.shape)
input_train_scaled = input_train_scaled.reshape(-1, 14, 15)
input_test_scaled = input_test_scaled.reshape(-1, 14, 15)
        
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
y_pred = modelq.predict(input_test_scaled)
y_pred = output_scaler.inverse_transform(y_pred)
mse = np.mean((target_test - y_pred) ** 2)
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
P_ = P
P_tensor = tf.constant(P_, dtype=tf.float32)
daily_vectors = []
differences = []
coal_costs = []
        
        
@app.route('/cost', methods=['POST'])
def cost():
    try:
        # Process request data
        data = request.json
        if not data:
            return jsonify({"error": "No data received in the request"}), 400
            
        # Extract coal blend data
        coal_blends = data.get("blends", [])
        if not coal_blends:
            return jsonify({"error": "No coal blend data provided"}), 400
            
        coal_types = [blend["coalType"] for blend in coal_blends]
        min_percentages = np.array([int(blend["minPercentage"]) for blend in coal_blends])
        max_percentages = np.array([int(blend["maxPercentage"]) for blend in coal_blends])
        
        # Validate data
        if len(coal_types) == 0:
            return jsonify({"error": "No coal types provided"}), 400
        
        # Pad arrays to fixed size of 14
        pad_size = 14 - len(min_percentages)
        min_percentages_padded = np.pad(min_percentages, (0, pad_size), mode='constant')
        max_percentages_padded = np.pad(max_percentages, (0, pad_size), mode='constant')
        
        # Extract desired coke parameters
        desired_coke_params = data.get("cokeParameters", {})
        if not desired_coke_params:
            return jsonify({"error": "No coke parameters provided"}), 400
        
        # Process blend coal data
        oneblends = data.get('blendcoal', [])
        user_input_values_padded = np.zeros(14)
        
        if oneblends:
            user_input_values = np.array([blend['currentRange'] for blend in oneblends])
            if abs(user_input_values.sum() - 100) > 0.001:  # Allow for floating point precision
                return jsonify({"error": "The total of current range must add up to 100."}), 400
            
            # Pad user input values
            user_input_values_padded[:len(user_input_values)] = user_input_values
            user_input_values_padded = user_input_values_padded.reshape(1, -1)
        
        # Get process type and parameters
        try:
            Option = int(data.get("processType"))
            if Option not in [1, 2, 3]:
                return jsonify({"error": f"Invalid option value: {Option}"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid process type: {data.get('processType')}"}), 400
        
        proces_para = data.get("processParameters", {})
        
        # Load process parameters based on option - use preloaded data if possible
        process_parameter_files = {
            1: 'Process_parameter_for_Rec_Top_Char.csv',
            2: 'Process_parameter_for_Rec_Stam_Char.csv',
            3: 'Process_parameter_for_Non_Rec_Stam_Char.csv'
        }
        
        Process_parameters = np.loadtxt(process_parameter_files[Option], delimiter=',')
        
        # Pad process parameters if needed
        if Option == 3:
            Process_parameters = np.pad(Process_parameters, ((0, 0), (0, 2)), mode='constant')
        
        # Combine matrices
        Conv_matrix = Blended_coal_parameters + Process_parameters
        
        # Split data for training - this could be precomputed if the model doesn't change
        X_train, X_test, y_train, y_test = train_test_split(Conv_matrix, Coke_properties, test_size=0.2, random_state=42)
        
        # Note: Keep the original dimensionality transformation pattern
        input_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # This should flatten to 210 features
        input_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        # Use existing scalers - the scalers expect 210 features (14*15 = 210)
        input_train_scaled = input__scaler.fit_transform(input_train_reshaped)
        input_test_scaled = input__scaler.transform(input_test_reshaped)
        
        target_train_scaled = output__scaler.fit_transform(y_train)
        
        # Generate combinations efficiently
        def generate_valid_combinations(min_vals, max_vals, num_coal_types, target_sum=100):
            """Generate valid coal blend combinations using a backtracking approach"""
            min_vals = min_vals[:num_coal_types].copy()
            max_vals = max_vals[:num_coal_types].copy()
            current = min_vals.copy()
            remaining = target_sum - current.sum()
            
            # Pre-allocate for better memory usage
            # Estimate max possible combinations to pre-allocate
            max_combinations = 10000  # Reasonable starting size
            valid_combinations = np.zeros((max_combinations, 14))
            count = 0
            
            def add_combination(combo):
                nonlocal count, valid_combinations
                # Resize array if needed
                if count >= len(valid_combinations):
                    valid_combinations = np.vstack([valid_combinations, np.zeros((max_combinations, 14))])
                
                valid_combinations[count, :num_coal_types] = combo
                count += 1
            
            def backtrack(idx, remaining):
                if idx == num_coal_types - 1:
                    # Last position - check if we can add the remaining amount
                    if min_vals[idx] <= remaining <= max_vals[idx]:
                        combo = current.copy()
                        combo[idx] = remaining
                        add_combination(combo)
                    return
                
                # Try values for current position - use adaptive step size for large ranges
                min_val = min_vals[idx]
                max_val = min(max_vals[idx], remaining - sum(min_vals[idx+1:]))
                
                # For large ranges, use a step size to reduce combinations
                range_size = max_val - min_val
                step = 1 if range_size < 50 else max(1, range_size // 50)
                
                for val in range(min_val, max_val + 1, step):
                    current[idx] = val
                    backtrack(idx + 1, remaining - val)
            
            # Start backtracking
            backtrack(0, remaining)
            
            # Return only the filled part of the array
            return valid_combinations[:count]
        
        # Generate combinations
        all_combinations = generate_valid_combinations(min_percentages_padded, max_percentages_padded, len(coal_types))
        
        # Cap combinations to prevent memory issues
        MAX_COMBINATIONS = 50000
        if len(all_combinations) > MAX_COMBINATIONS:
            # Sample combinations if too many
            indices = np.random.choice(len(all_combinations), MAX_COMBINATIONS, replace=False)
            all_combinations = all_combinations[indices]
        
        # Process parameters padding
        if isinstance(proces_para, dict):
            # Convert dictionary to array based on expected keys
            # This part depends on your exact data structure
            proces_para = np.array([proces_para.get(f'param{i}', 0) for i in range(1, 16)])
        
        if Option == 3:
            proces_para = np.pad(proces_para, (0, 2), mode='constant')
        
        # TensorFlow operations - batch processing
        # Convert to constants once
        D_tensor = tf.constant(all_combinations, dtype=tf.float32)
        
        # Reshape for broadcasting
        D_expanded = tf.expand_dims(D_tensor, 2)  # shape: [n, d, 1]
        P_expanded = tf.expand_dims(P_tensor, 0)  # shape: [1, d, m]
        
        # Compute daily vectors efficiently
        daily_vectors = tf.multiply(D_expanded, P_expanded)
        daily_vectors = tf.transpose(daily_vectors, perm=[0, 2, 1])
        
        # Important: Keep the original reshape pattern to match the expected 210 features
        b1 = daily_vectors.numpy().reshape(daily_vectors.shape[0], -1)  # This flattens to 210 features
        
        # Keep the original model input shape
        b1_reshaped = daily_vectors.numpy().reshape(-1, 14, 15)
        
        # Predict in batches if needed
        BATCH_SIZE = 1000
        blend1 = []
        
        for i in range(0, len(b1_reshaped), BATCH_SIZE):
            batch = b1_reshaped[i:i+BATCH_SIZE]
            batch_pred = modelq.predict(batch, verbose=0)  # Silence predictions
            blend1.append(batch_pred)
            
        blend1 = np.vstack(blend1)
        blended_coal_properties = output_scaler.inverse_transform(blend1)
        
        # Process for coke prediction - use broadcasting for addition
        proces_para_broadcast = np.broadcast_to(proces_para, (blend1.shape[0], proces_para.shape[0]))
        blend1_with_process = blend1 + proces_para_broadcast
        
        # Flatten and scale - crucial to reshape to match the expected 210 features
        blend1_flattened = blend1_with_process.reshape(blend1_with_process.shape[0], -1)
        
        # Scale using the scaler that expects 210 features
        blend1_scaled = input__scaler.transform(blend1_flattened)
        
        # Predict coke properties in batches
        predictions = []
        for i in range(0, len(blend1_scaled), BATCH_SIZE):
            batch = blend1_scaled[i:i+BATCH_SIZE]
            batch_pred = rf_model.predict(batch, verbose=0)  # Silence predictions
            predictions.append(batch_pred)
            
        predictions = np.vstack(predictions)
        predictions = output__scaler.inverse_transform(predictions)
        
        # Extract desired parameters
        desired_params = {
            "ash": desired_coke_params.get("ASH", 0),
            "vm": desired_coke_params.get("VM", 0),
            "m40": desired_coke_params.get("M_40MM", 0),
            "m10": desired_coke_params.get("M_10MM", 0),
            "csr": desired_coke_params.get("CSR", 0),
            "cri": desired_coke_params.get("CRI", 0),
            "ams": desired_coke_params.get("AMS", 0)
        }
        
        # Validate desired parameters
        if any(value == 0 for value in desired_params.values()):
            return jsonify({"error": "Missing or zero value in coke parameters"}), 400
        
        # Filter valid predictions - use vectorized operations
        def filter_valid_predictions(predictions, combinations, blended_properties, min_max_values):
            """Filter predictions that meet all constraints using vectorized operations"""
            
            # Create masks for each constraint simultaneously
            masks = np.array([
                (min_max_values['ash']['lower'] <= predictions[:, 0]) & (predictions[:, 0] <= min_max_values['ash']['upper']),
                (min_max_values['vm']['lower'] <= predictions[:, 1]) & (predictions[:, 1] <= min_max_values['vm']['upper']),
                (min_max_values['m40']['lower'] <= predictions[:, 9]) & (predictions[:, 9] <= min_max_values['m40']['upper']),
                (min_max_values['m10']['lower'] <= predictions[:, 10]) & (predictions[:, 10] <= min_max_values['m10']['upper']),
                (min_max_values['csr']['lower'] <= predictions[:, 12]) & (predictions[:, 12] <= min_max_values['csr']['upper']),
                (min_max_values['cri']['lower'] <= predictions[:, 13]) & (predictions[:, 13] <= min_max_values['cri']['upper']),
                (min_max_values['ams']['lower'] <= predictions[:, 14]) & (predictions[:, 14] <= min_max_values['ams']['upper'])
            ])
            
            # Combine all masks with logical AND
            valid_mask = np.all(masks, axis=0)
            
            # Get valid indices
            valid_indices = np.where(valid_mask)[0]
            invalid_indices = np.where(~valid_mask)[0]
            
            # Return valid and invalid data
            return (
                predictions[valid_indices],
                combinations[valid_indices],
                [blended_properties[i] for i in valid_indices],
                predictions[invalid_indices],
                combinations[invalid_indices],
                [blended_properties[i] for i in invalid_indices]
            )
        
        # Filter valid predictions
        (valid_predictions, valid_combinations, valid_blended_coal_properties,
         invalid_predictions, invalid_combinations, invalid_blended_coal_properties) = filter_valid_predictions(
            predictions, all_combinations, blended_coal_properties, min_max_values)
        
        # If no valid predictions, return error
        if len(valid_predictions) == 0:
            return jsonify({
                "error": "No valid coal blends found that meet the specified criteria.",
                "total_combinations_tried": len(all_combinations),
                "suggestion": "Consider relaxing your constraints or increasing the range of allowable percentages."
            }), 400
        
        # Extract relevant columns for difference calculation
        pred_indices = [0, 1, 9, 10, 12, 13, 14]  # ASH, VM, M40, M10, CSR, CRI, AMS
        pred_cols = valid_predictions[:, pred_indices]
        
        # Target values array
        target_values = np.array([
            desired_params["ash"],
            desired_params["vm"],
            desired_params["m40"],
            desired_params["m10"],
            desired_params["csr"],
            desired_params["cri"],
            desired_params["ams"]
        ])
        
        # Weights array
        weights = np.array([
            min_max_values['ash']['weight'],
            min_max_values['vm']['weight'],
            min_max_values['m40']['weight'],
            min_max_values['m10']['weight'],
            min_max_values['csr']['weight'],
            min_max_values['cri']['weight'],
            min_max_values['ams']['weight']
        ])
        
        # Create masks for better/worse metrics
        # Lower is better for ASH(0), VM(1), M10(3), CRI(5)
        # Higher is better for M40(2), CSR(4), AMS(6)
        lower_better_mask = np.array([True, True, False, True, False, True, False])
        
        # Calculate percentage differences with vectorized operations
        # Broadcast target_values to match pred_cols shape
        targets_broadcast = np.tile(target_values, (len(pred_cols), 1))
        
        # Calculate differences based on whether higher/lower is better
        diff_matrix = np.zeros_like(pred_cols)
        
        # For lower is better metrics
        lower_indices = np.where(lower_better_mask)[0]
        diff_matrix[:, lower_indices] = (targets_broadcast[:, lower_indices] - pred_cols[:, lower_indices]) / targets_broadcast[:, lower_indices]
        
        # For higher is better metrics
        higher_indices = np.where(~lower_better_mask)[0]
        diff_matrix[:, higher_indices] = (pred_cols[:, higher_indices] - targets_broadcast[:, higher_indices]) / targets_broadcast[:, higher_indices]
        
        # Apply weights with broadcasting
        weighted_diffs = diff_matrix * weights
        
        # Calculate total differences
        total_differences = np.sum(weighted_diffs, axis=1)
        
        # Sort by performance (descending)
        sorted_performance_indices = np.argsort(total_differences)[::-1]
        
        # Apply sorted indices
        sorted_predictions = valid_predictions[sorted_performance_indices]
        sorted_blends = valid_combinations[sorted_performance_indices]
        sorted_blended_coal_properties = [valid_blended_coal_properties[i] for i in sorted_performance_indices]
        
        # Calculate costs with vectorized operations
        coal_cost_map = {row[0]: float(row[-2]) for row in data1.itertuples(index=False)}
        
        # Get costs for each coal type
        coal_costs = np.array([coal_cost_map.get(coal_type, 0.0) for coal_type in coal_types])
        
        # Vectorized cost calculation using dot product
        # Only use the relevant part of each blend that corresponds to actual coal types
        total_costs = np.zeros(len(valid_combinations))
        for i, blend in enumerate(valid_combinations):
            # Use only the part of the blend that corresponds to actual coal types
            total_costs[i] = np.dot(blend[:len(coal_types)], coal_costs) / 100
        
        # Sort by cost (ascending)
        sorted_indices_by_cost = np.argsort(total_costs)
        
        sorted_blend_cost = valid_combinations[sorted_indices_by_cost]
        sorted_prediction_cost = valid_predictions[sorted_indices_by_cost]
        sorted_total_cost = total_costs[sorted_indices_by_cost]
        sorted_blended_coal_properties_cost = [valid_blended_coal_properties[i] for i in sorted_indices_by_cost]
        
        # Calculate combined score
        normalized_costs = np.zeros_like(total_costs)
        if len(total_costs) > 0 and np.max(total_costs) > np.min(total_costs):
            normalized_costs = (total_costs - np.min(total_costs)) / (np.max(total_costs) - np.min(total_costs))
            
        normalized_differences = np.zeros_like(total_differences)
        if len(total_differences) > 0 and np.max(total_differences) > np.min(total_differences):
            normalized_differences = ((total_differences - np.min(total_differences)) / 
                                   (np.max(total_differences) - np.min(total_differences)))
        
        # Get weights for combined score
        cost_weight = min_max_values.get('cost_weightage', 0.5)
        performance_weight = min_max_values.get('coke_quality', 0.5)
        
        # Ensure min_len is handled properly
        combined_scores = (cost_weight * normalized_costs) + (performance_weight * normalized_differences)
        
        # Find best combined score (minimum)
        best_combined_index = np.argmin(combined_scores)
        
        # Select three representative blends
        # 1. Best by performance (highest difference score)
        best_performance_index = sorted_performance_indices[0]
        blend_1 = valid_combinations[best_performance_index]
        blended_coal_1 = valid_blended_coal_properties[best_performance_index]
        blend_1_properties = valid_predictions[best_performance_index]
        blend_1_cost = total_costs[best_performance_index]
        
        # 2. Cheapest
        cheapest_index = sorted_indices_by_cost[0]
        blend_2 = valid_combinations[cheapest_index]
        blended_coal_2 = valid_blended_coal_properties[cheapest_index]
        blend_2_properties = valid_predictions[cheapest_index]
        blend_2_cost = total_costs[cheapest_index]
        
        # 3. Best combined
        blend_3 = valid_combinations[best_combined_index]
        blended_coal_3 = valid_blended_coal_properties[best_combined_index]
        blend_3_properties = valid_predictions[best_combined_index]
        blend_3_cost = total_costs[best_combined_index]
        
        # Trim zero-padding from results for cleaner output
        def clean_output(array, actual_length):
            if isinstance(array, list):
                return array[:actual_length]
            if isinstance(array, np.ndarray):
                return array[:actual_length].tolist()
            return array
            
        num_coal_types = len(coal_types)
        
        # Prepare response
        response = {
            "blend1": {
                "composition": clean_output(blend_1, num_coal_types),
                "blendedcoal": blended_coal_1,
                "properties": blend_1_properties.tolist(),
                "cost": float(blend_1_cost)
            },
            "blend2": {
                "composition": clean_output(blend_2, num_coal_types),
                "blendedcoal": blended_coal_2,
                "properties": blend_2_properties.tolist(),
                "cost": float(blend_2_cost)
            },
            "blend3": {
                "composition": clean_output(blend_3, num_coal_types),
                "blendedcoal": blended_coal_3,
                "properties": blend_3_properties.tolist(),
                "cost": float(blend_3_cost)
            },
            "valid_predictions_count": len(valid_predictions),
            "total_combinations": len(all_combinations)
        }
        
        # Process user input if provided
        if np.any(user_input_values_padded != 0):
            # Process user custom blend - maintains original reshape pattern to match 210 features
            D_tensor = tf.constant(user_input_values_padded, dtype=tf.float32)
            
            # Calculate product directly
            user_daily_vectors = tf.multiply(
                tf.expand_dims(D_tensor, 2),
                tf.expand_dims(P_tensor, 0)
            )
            user_daily_vectors = tf.transpose(user_daily_vectors, perm=[0, 2, 1])
            
            # Reshape for prediction - maintain 14x15 shape for modelq
            user_vectors_reshaped = user_daily_vectors.numpy().reshape(-1, 14, 15)
            
            # Predict blended coal properties
            user_prediction_scaled = modelq.predict(user_vectors_reshaped, verbose=0)
            user_prediction = output_scaler.inverse_transform(user_prediction_scaled)
            
            # Add process parameters
            proces_para_broadcast = np.broadcast_to(proces_para, user_prediction.shape)
            user_conv = proces_para_broadcast + user_prediction
            
            # Critical: reshape to match the expected 210 features for input__scaler
            user_conv_flattened = user_conv.reshape(user_conv.shape[0], -1)
            user_conv_scaled = input__scaler.transform(user_conv_flattened)
            
            # Predict coke properties
            user_coke = rf_model.predict(user_conv_scaled, verbose=0)
            user_predictions = output__scaler.inverse_transform(user_coke)
            
            # Calculate cost for custom blend
            user_blend_cost = sum(user_input_values_padded[0, i] * coal_cost_map.get(coal_types[i], 0) / 100 
                               for i in range(min(user_input_values_padded.shape[1], len(coal_types))))
            
            # Add to response
            response["ProposedCoal"] = {
                "BlendProperties": user_prediction.tolist(),
                "CokeProperties": user_predictions.tolist(),
                "Cost": float(user_blend_cost)
            }
        else:
            response["ProposedCoal"] = {
                "message": "No custom blend data provided."
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        # Enhanced error handling with stack trace for debugging
        trace = traceback.format_exc()
        app.logger.error(f"Error in cost endpoint: {str(e)}\n{trace}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
#coal properties page 

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



