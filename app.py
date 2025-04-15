import pandas as pd
import numpy as np
import csv
import os
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


@app.route('/cost', methods=['POST'])
def cost():
    
        data = request.json  
        if not data:
            raise ValueError("No data received in the request") 
        coal_blends = data.get("blends")
        coal_types = [blk["coalType"] for blk in coal_blends]
        min_percentages = [int(blk["minPercentage"]) for blk in coal_blends]
        max_percentages = [int(blk["maxPercentage"]) for blk in coal_blends]
        
        min_percentages_padded = np.pad(min_percentages, (0, 14 - len(min_percentages)), mode='constant') 
        max_percentages_padded = np.pad(max_percentages, (0, 14 - len(max_percentages)), mode='constant')
        desired_coke_params = data.get("cokeParameters")
        
        print("desired coke Parameters:", desired_coke_params)
        oneblends = data.get('blendcoal', [])
        user_input_values_padded = np.zeros(14)

        if oneblends:
            user_input_values = np.array([blend['currentRange'] for blend in oneblends])
            if user_input_values.sum() != 100:
                return jsonify({"error": "The total of current range must add up to 100."}), 400
            user_input_values_padded = np.pad(user_input_values, (0, 14 - len(user_input_values)), mode='constant')
            user_input_values_padded = np.array(user_input_values_padded)
            user_input_values_padded = np.array(user_input_values_padded).reshape(1, -1)
            print("D-tensor-1", user_input_values_padded)
            
        else:
            # If blendcoal data is missing, don't do anything (leave the padded zero array)
            pass
        
        print("D-tensor", user_input_values_padded)
        Option = data.get("processType")
        print(f"Option (type: {type(Option)}):", Option)
        
        try:
            Option = int(Option) 
        except ValueError:
            raise ValueError(f"Invalid option value: {Option} (could not convert to integer)")
        
        proces_para= data.get("processParameters", {})
        
        print("model Process Parameters:", proces_para)

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
        # coke_properties
        Coke_properties = np.loadtxt('coke_properties.csv', delimiter=',')
        
        data = pd.read_csv('individual_coal_prop.csv', dtype=str,header=None, on_bad_lines='skip')

        I = np.loadtxt('individual_coal_prop.csv', delimiter=',', usecols=range(1, data.shape[1] - 2))
        

        if Option == 1:
            Process_parameters = np.loadtxt('Process_parameter_for_Rec_Top_Char.csv', delimiter=',')
        elif Option == 2:
            Process_parameters = np.loadtxt('Process_parameter_for_Rec_Stam_Char.csv', delimiter=',')
        elif Option == 3:
            Process_parameters = np.loadtxt('Process_parameter_for_Non_Rec_Stam_Char.csv', delimiter=',')
            print("This was running")
        else:
            raise ValueError(f"Invalid option value: {Option}")
            
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
        
        # Scaling
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
        
        
        modelq.fit(input_train_scaled, target_train_scaled, epochs=100, batch_size=8, validation_data=(input_test_scaled, target_test_scaled))
        y_pred = modelq.predict(input_test_scaled)
        y_pred = output_scaler.inverse_transform(y_pred)
        mse = np.mean((target_test - y_pred) ** 2)
        
        if Option == 3:
            Process_parameters = np.pad(Process_parameters, ((0, 0), (0, 2)), mode='constant', constant_values=0)
        
        Conv_matrix = Blended_coal_parameters + Process_parameters
        
        X_train, X_test, y_train, y_test = train_test_split(Conv_matrix, Coke_properties, test_size=0.2, random_state=42)
        
        # Scaling second phase
        
        input__scaler = MinMaxScaler()
        output__scaler = MinMaxScaler()
        input_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        input_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        input_train_scaled = input__scaler.fit_transform(input_train_reshaped)
        input_test_scaled = input__scaler.transform(input_test_reshaped)
        
        target_train_scaled = output__scaler.fit_transform(y_train)
        target_test_scaled = output__scaler.transform(y_test)
        # # Define second model
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
        rf_model.fit(input_train_scaled, target_train_scaled, epochs=100, batch_size=8, validation_data=(input_test_scaled, target_test_scaled))
        
        
        
        y_pred = rf_model.predict(input_test_scaled)
        y_pred = output_scaler.inverse_transform(y_pred)
        mse = np.mean((y_test - y_pred) ** 2)
        
    
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
                    

        all_combinations = np.array(list(generate_combinations(0, [], 0)))
        
        if(Option==3):
            proces_para = np.pad(proces_para, (0,2), mode='constant', constant_values=0)
            
        D_ = all_combinations
        P_ = P
        
        # Compute daily vectors
        D_tensor = tf.constant(D_, dtype=tf.float32)
        P_tensor = tf.constant(P_, dtype=tf.float32)
        
        daily_vectors = []
        for i in range(D_tensor.shape[0]):
            row_vector = []
            for j in range(P_tensor.shape[1]):
                product_vector = tf.multiply(D_tensor[i], P_tensor[:, j])
                row_vector.append(product_vector)
            daily_vectors.append(tf.stack(row_vector))
        
        daily_vectors_tensor = tf.stack(daily_vectors)
        input_data = tf.reshape(daily_vectors_tensor, [-1, 14])
        
        daily_vectors_flattened = daily_vectors_tensor.numpy().reshape(daily_vectors_tensor.shape[0], -1)
        b1=daily_vectors_flattened
        
        
        
        b1= b1.reshape(b1.shape[0], -1)
        b1_scaled = input_scaler.transform(b1)
        b1 = b1.reshape(-1, 14, 15)
        blend1=modelq.predict(b1)
        blended_coal_properties=output_scaler.inverse_transform(blend1)
        blend1=blend1+proces_para
        blend1 = blend1.reshape(blend1.shape[0], -1)
        blend1 = input__scaler.transform(blend1)
        coke = rf_model.predict(blend1)
        predictions=output__scaler.inverse_transform(coke)
        
        
        
        def read_min_max_values():
            df = pd.read_csv('min-maxvalues.csv')
            print(df);
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
        
       
        desired_ash = desired_coke_params["ASH"]
        desired_vm = desired_coke_params["VM"]
        desired_m40 = desired_coke_params["M_40MM"]
        desired_m10 = desired_coke_params["M_10MM"]
        desired_csr = desired_coke_params["CSR"]
        desired_cri = desired_coke_params["CRI"]
        desired_ams = desired_coke_params["AMS"]
        
        def filter_valid_and_invalid(predictions, combinations, blended_coal_properties,min_max_values):
            ash_min = min_max_values['ash']['lower']
            ash_max = min_max_values['ash']['upper']
            vm_min = min_max_values['vm']['lower']
            vm_max= min_max_values['vm']['upper']
            m40_min = min_max_values['m40']['lower']
            m40_max = min_max_values['m40']['upper']
            m10_min = min_max_values['m10']['lower']
            m10_max = min_max_values['m10']['upper']
            csr_min =  min_max_values['csr']['lower']
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
                # Check if all values are within the specified range
                if (
                    ash_min <= prediction[0] <= ash_max and  # ASH
                    vm_min <= prediction[1] <= vm_max and  # VM
                    m40_min <= prediction[9] <= m40_max and  # M_40
                    m10_min <= prediction[10] <= m10_max and  # M_10
                    csr_min <= prediction[12] <= csr_max and  # CSR
                    cri_min <= prediction[13] <= cri_max  and# CRI
                    ams_min <= prediction[14] <= ams_max   # AMS
                    
                ):
                    valid_indices.append(i)
                else:
                    invalid_indices.append(i)
            # Separate valid and invalid predictions, combinations, and blended coal properties
            valid_predictions = predictions[valid_indices]
            valid_combinations = combinations[valid_indices]
            valid_blended_coal_properties = [blended_coal_properties[i] for i in valid_indices]
            invalid_predictions = predictions[invalid_indices]
            invalid_combinations = combinations[invalid_indices]
            invalid_blended_coal_properties = [blended_coal_properties[i] for i in invalid_indices]
            
            print(f"Number of valid predictions: {len(valid_predictions)}")
            print(f"Number of invalid predictions: {len(invalid_predictions)}")


            return (
                valid_predictions,
                valid_combinations,
                valid_blended_coal_properties,
                invalid_predictions,
                invalid_combinations,
                invalid_blended_coal_properties,
            )

# Filtering valid and invalid predictions, combinations, and blended coal properties
        (
            valid_predictions,
            valid_combinations,
            valid_blended_coal_properties,
            invalid_predictions,
            invalid_combinations,
            invalid_blended_coal_properties,
        ) = filter_valid_and_invalid(predictions, all_combinations, blended_coal_properties, min_max_values)
        
        predictions = valid_predictions
        all_combinations = valid_combinations
        blended_coal_properties = valid_blended_coal_properties
        
        

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
        
                
        coal_costs = []
        for i, blend in enumerate(sorted_blends):
            coal_type_costs = []
            for j, coal_type in enumerate(coal_types):
                if j < len(blend):
                    # Map the coal type to the CSV file and retrieve the cost
                    coal_type_cost = float(data.loc[data[0] == coal_type, data.columns[-2]].values[0])
                    coal_type_costs.append(coal_type_cost)
            coal_costs.append(coal_type_costs)

        total_costs = [sum(float(blend[i]) * coal_costs[j][i] / 100 for i in range(min(len(blend), len(coal_costs[j])))) for j, blend in enumerate(sorted_blends)] 
       
       
       
        print(f"coal_costs: {coal_costs}")
        
        sorted_indices_by_cost = np.argsort(total_costs)
        sorted_blend_cost = sorted_blends[sorted_indices_by_cost]
        sorted_prediction_cost = sorted_predictions[sorted_indices_by_cost]
        sorted_total_cost = np.array(total_costs)[sorted_indices_by_cost]

        sorted_blended_coal_properties_cost = [sorted_blended_coal_properties[i] for i in sorted_indices_by_cost]
        sorted_diff_cost = [sorted_diff[i] for i in sorted_indices_by_cost]

        # -----------------------------------------------------------------------------
        # Combine Cost and Performance
        # -----------------------------------------------------------------------------
        normalized_costs = (total_costs - np.min(total_costs)) / (np.max(total_costs) - np.min(total_costs))
        normalized_differences = (
            (total_differences - np.min(total_differences))
            / (np.max(total_differences) - np.min(total_differences))
        )

        cost_weight = min_max_values['cost_weightage']
        performance_weight = min_max_values['coke_quality']

        combined_scores = (cost_weight * normalized_costs) + (performance_weight * normalized_differences)

        # best_combined_index refers to the ORIGINAL arrays (not sorted)
        best_combined_index = np.argmin(combined_scores)

        # -----------------------------------------------------------------------------
        # Helper for Cost Calculation
        # -----------------------------------------------------------------------------
        def calculate_cost(blend, coal_costs):
            return sum(blend[i] * coal_costs[0][i] / 100 for i in range(min(len(blend), len(coal_costs[0]))))

        # =============================================================================
        # 3. Pick Three Representative Blends
        # =============================================================================

        # 3.1. Best by performance (first in performance-sorted list)
        blend_1 = sorted_blends[0]
        blended_coal_1 = sorted_blended_coal_properties[0]
        blend_1_properties = sorted_predictions[0]
        blend_1_cost = calculate_cost(blend_1, coal_costs)

        # 3.2. Cheapest (first in cost-sorted list)
        blend_2 = sorted_blend_cost[0]
        blended_coal_2 = sorted_blended_coal_properties_cost[0]
        blend_2_properties = sorted_prediction_cost[0]
        blend_2_cost = sorted_total_cost[0]  # Already computed above

        # 3.3. Best combined (from original arrays, using best_combined_index)
        # Fix 2: Use original arrays so the index lines up.
        blend_3 = all_combinations[best_combined_index]
        blended_coal_3 = valid_blended_coal_properties[best_combined_index]
        blend_3_properties = valid_predictions[best_combined_index]
        blend_3_cost = calculate_cost(blend_3, coal_costs)
        
        print(f"Blend 1 Cost: {blend_1_cost}")
        print(f"Blend 2 Cost: {blend_2_cost}")
        print(f"Blend 3 Cost: {blend_3_cost}")
                
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
        
        if np.any(user_input_values_padded != 0):
            D_tensor = tf.constant(user_input_values_padded, dtype=tf.float32)
            print(D_tensor)
            daily_vector_test = []
            D_test = tf.constant(D_tensor)
            P_test = tf.constant(P_tensor)


            daily_vectors = []
            for i in range(D_tensor.shape[0]):
                row_vector = []
                for j in range(P_tensor.shape[1]):
                    product_vector = tf.multiply(tf.cast(D_tensor[i], tf.float32), tf.cast(P_tensor[:, j], tf.float32))
                    row_vector.append(product_vector)
                daily_vectors.append(tf.stack(row_vector))
            daily_vectors_tensor = tf.stack(daily_vectors)
            input_data = tf.reshape(daily_vectors_tensor, [-1, 14])


            daily_vectors_tensor.shape
            daily_vectors_tensor_test=daily_vectors_tensor
            daily_vectors_tensor_test_reshaped = daily_vectors_tensor_test.numpy().reshape(1, -1)
            
            daily_vectors_tensor_test_scaled = input_scaler.transform(daily_vectors_tensor_test_reshaped)
            print("Shape before reshaping:", daily_vectors_tensor_test_scaled.shape)
            daily_vectors_tensor_test_scaled = daily_vectors_tensor_test_scaled.reshape(-1, 14, 15)
            prediction_scaled = modelq.predict(daily_vectors_tensor_test_scaled)
            prediction = output_scaler.inverse_transform(prediction_scaled)

            print("Predicted values:", prediction)
            
            Conv =proces_para+prediction
            # blend1 = blend1.reshape(blend1.shape[0], -1)
            blend1 = input__scaler.transform(Conv)
            coke = rf_model.predict(Conv)
            predictions=output__scaler.inverse_transform(coke)
            
            print(predictions)
            
            # Add the predicted proposed coal and coke values to the response
            response["ProposedCoal"] = {
                "Blend2": prediction.tolist(),
                "Coke2": predictions.tolist()
            }
        else:
        # If no valid blendcoal data is provided, indicate that no prediction is made
            response["ProposedCoal"] = {
                "error": "No valid blendcoal data provided, unable to make predictions."
            }       
        return jsonify(response), 200
# Helper functions
def create_nn_model(input_shape, output_size):
    """Create a neural network model with the specified input shape and output size."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
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
        layers.Dense(output_size, activation='linear')
    ])
    return model

def generate_all_combinations(min_percentages, max_percentages):
    """Generate all valid combinations of coal percentages adding up to 100%."""
    def generate_combinations(index, current_combination, current_sum):
        target_sum = 100
        if index == len(min_percentages) - 1:
            remaining = target_sum - current_sum
            if min_percentages[index] <= remaining <= max_percentages[index]:
                yield current_combination + [remaining]
            return
        for value in range(min_percentages[index], max_percentages[index] + 1):
            if current_sum + value <= target_sum:
                yield from generate_combinations(index + 1, current_combination + [value], current_sum + value)
    
    return np.array(list(generate_combinations(0, [], 0)))

def compute_daily_vectors(D_, P_):
    """Compute daily vectors using tensor operations."""
    D_tensor = tf.constant(D_, dtype=tf.float32)
    P_tensor = tf.constant(P_, dtype=tf.float32)
    
    daily_vectors = []
    for i in range(D_tensor.shape[0]):
        row_vectors = [tf.multiply(D_tensor[i], P_tensor[:, j]) for j in range(P_tensor.shape[1])]
        daily_vectors.append(tf.stack(row_vectors))
    
    return tf.stack(daily_vectors)

def read_min_max_values():
    """Read min-max values from CSV file."""
    df = pd.read_csv('min-maxvalues.csv')
    return {
        'ash': {'lower': df['ash_lower'].iloc[0], 'upper': df['ash_upper'].iloc[0], 'weight': df['ash_weight'].iloc[0]},
        'vm': {'lower': df['vm_lower'].iloc[0], 'upper': df['vm_upper'].iloc[0], 'weight': df['vm_weight'].iloc[0]},
        'm40': {'lower': df['m40_lower'].iloc[0], 'upper': df['m40_upper'].iloc[0], 'weight': df['m40_weight'].iloc[0]},
        'm10': {'lower': df['m10_lower'].iloc[0], 'upper': df['m10_upper'].iloc[0], 'weight': df['m10_weight'].iloc[0]},
        'csr': {'lower': df['csr_lower'].iloc[0], 'upper': df['csr_upper'].iloc[0], 'weight': df['csr_weight'].iloc[0]},
        'cri': {'lower': df['cri_lower'].iloc[0], 'upper': df['cri_upper'].iloc[0], 'weight': df['cri_weight'].iloc[0]},
        'ams': {'lower': df['ams_lower'].iloc[0], 'upper': df['ams_upper'].iloc[0], 'weight': df['ams_weight'].iloc[0]},
        'cost_weightage': df['cost_weightage'].iloc[0],
        'coke_quality': df['coke_quality'].iloc[0]
    }

def extract_desired_params(desired_coke_params):
    """Extract desired coke parameters from input data."""
    return {
        'ash': desired_coke_params.get("ASH", 0),
        'vm': desired_coke_params.get("VM", 0),
        'm40': desired_coke_params.get("M_40MM", 0),
        'm10': desired_coke_params.get("M_10MM", 0),
        'csr': desired_coke_params.get("CSR", 0),
        'cri': desired_coke_params.get("CRI", 0),
        'ams': desired_coke_params.get("AMS", 0)
    }

def filter_valid_predictions(predictions, combinations, blended_coal_properties, min_max_values):
    """Filter valid and invalid predictions based on min-max criteria."""
    valid_indices = []
    
    for i, prediction in enumerate(predictions):
        # Check if all values are within the specified range
        if (min_max_values['ash']['lower'] <= prediction[0] <= min_max_values['ash']['upper'] and  # ASH
            min_max_values['vm']['lower'] <= prediction[1] <= min_max_values['vm']['upper'] and    # VM
            min_max_values['m40']['lower'] <= prediction[9] <= min_max_values['m40']['upper'] and  # M_40
            min_max_values['m10']['lower'] <= prediction[10] <= min_max_values['m10']['upper'] and # M_10
            min_max_values['csr']['lower'] <= prediction[12] <= min_max_values['csr']['upper'] and # CSR
            min_max_values['cri']['lower'] <= prediction[13] <= min_max_values['cri']['upper'] and # CRI
            min_max_values['ams']['lower'] <= prediction[14] <= min_max_values['ams']['upper']):   # AMS
            valid_indices.append(i)
    
    # Extract valid predictions, combinations, and blended coal properties
    valid_predictions = predictions[valid_indices]
    valid_combinations = combinations[valid_indices]
    valid_blended_coal_properties = [blended_coal_properties[i] for i in valid_indices]
    
    return (valid_predictions, valid_combinations, valid_blended_coal_properties)

def calculate_differences(predictions, desired_params, min_max_values):
    """Calculate weighted differences between predicted and desired parameters."""
    differences = []
    for prediction in predictions:
        diff = [
            ((desired_params['ash'] - prediction[0]) / desired_params['ash']) * min_max_values['ash']['weight'],
            ((desired_params['vm'] - prediction[1]) / desired_params['vm']) * min_max_values['vm']['weight'],
            ((prediction[9] - desired_params['m40']) / desired_params['m40']) * min_max_values['m40']['weight'],
            ((desired_params['m10'] - prediction[10]) / desired_params['m10']) * min_max_values['m10']['weight'],
            ((prediction[12] - desired_params['csr']) / desired_params['csr']) * min_max_values['csr']['weight'],
            ((desired_params['cri'] - prediction[13]) / desired_params['cri']) * min_max_values['cri']['weight'],
            ((prediction[14] - desired_params['ams']) / desired_params['ams']) * min_max_values['ams']['weight']
        ]
        differences.append(diff)
    
    total_differences = [sum(diff) for diff in differences]
    return differences, total_differences

def calculate_coal_costs(coal_types, blends, coal_data_df):
    """Calculate coal costs for each blend."""
    coal_costs = []
    for blend in blends:
        coal_type_costs = []
        for j, coal_type in enumerate(coal_types):
            if j < len(blend):
                # Get cost from CSV data
                coal_type_cost = float(coal_data_df.loc[coal_data_df[0] == coal_type, coal_data_df.columns[-2]].values[0])
                coal_type_costs.append(coal_type_cost)
        coal_costs.append(coal_type_costs)
    return coal_costs

def calculate_total_costs(blends, coal_costs):
    """Calculate total cost for each blend."""
    return [sum(float(blend[i]) * coal_costs[j][i] / 100 
               for i in range(min(len(blend), len(coal_costs[j])))) 
            for j, blend in enumerate(blends)]

def calculate_combined_scores(costs, differences, min_max_values):
    """Calculate combined scores based on normalized costs and differences."""
    normalized_costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs)) if np.max(costs) != np.min(costs) else np.zeros_like(costs)
    normalized_differences = (differences - np.min(differences)) / (np.max(differences) - np.min(differences)) if np.max(differences) != np.min(differences) else np.zeros_like(differences)
    
    cost_weight = min_max_values['cost_weightage']
    performance_weight = min_max_values['coke_quality']
    
    return (cost_weight * normalized_costs) + (performance_weight * normalized_differences)

def calculate_cost(blend, coal_costs):
    """Calculate cost for a single blend."""
    return sum(blend[i] * coal_costs[0][i] / 100 for i in range(min(len(blend), len(coal_costs[0]))))

def select_best_blends(sorted_blends, sorted_blended_coal_properties, sorted_predictions,
                       sorted_blend_cost, sorted_blended_coal_properties_cost, sorted_prediction_cost,
                       sorted_total_cost, valid_data, best_combined_index, coal_costs):
    """Select three representative blends: best performance, cheapest, and best combined."""
    # Best by performance
    blend_1 = {
        "composition": sorted_blends[0],
        "blendedcoal": sorted_blended_coal_properties[0],
        "properties": sorted_predictions[0],
        "cost": calculate_cost(sorted_blends[0], coal_costs)
    }
    
    # Cheapest
    blend_2 = {
        "composition": sorted_blend_cost[0],
        "blendedcoal": sorted_blended_coal_properties_cost[0],
        "properties": sorted_prediction_cost[0],
        "cost": sorted_total_cost[0]
    }
    
    # Best combined
    blend_3 = {
        "composition": valid_data[1][best_combined_index],
        "blendedcoal": valid_data[2][best_combined_index],
        "properties": valid_data[0][best_combined_index],
        "cost": calculate_cost(valid_data[1][best_combined_index], coal_costs)
    }
    
    return {"blend1": blend_1, "blend2": blend_2, "blend3": blend_3}

def create_response_with_blends(best_blends, valid_count):
    """Create API response with the three best blends."""
    return {
        "blend1": {
            "composition": best_blends["blend1"]["composition"].tolist(),
            "blendedcoal": best_blends["blend1"]["blendedcoal"].tolist(),
            "properties": best_blends["blend1"]["properties"].tolist(),
            "cost": best_blends["blend1"]["cost"]
        },
        "blend2": {
            "composition": best_blends["blend2"]["composition"].tolist(),
            "blendedcoal": best_blends["blend2"]["blendedcoal"].tolist(),
            "properties": best_blends["blend2"]["properties"].tolist(),
            "cost": best_blends["blend2"]["cost"]
        },
        "blend3": {
            "composition": best_blends["blend3"]["composition"].tolist(),
            "blendedcoal": best_blends["blend3"]["blendedcoal"].tolist(),
            "properties": best_blends["blend3"]["properties"].tolist(),
            "cost": best_blends["blend3"]["cost"]
        },
        "valid_predictions_count": valid_count
    }

def predict_user_blend(user_input, P_tensor, input_scaler, modelq, output_scaler, 
                      process_params, input__scaler, rf_model, output__scaler):
    """Predict properties for user-provided blend."""
    # Compute daily vectors for user input
    daily_vectors = []
    for i in range(user_input.shape[0]):
        row_vectors = [tf.multiply(tf.cast(user_input[i], tf.float32), 
                                  tf.cast(P_tensor[:, j], tf.float32)) 
                      for j in range(P_tensor.shape[1])]
        daily_vectors.append(tf.stack(row_vectors))
    
    daily_vectors_tensor = tf.stack(daily_vectors)
    
    # Reshape and scale
    daily_vectors_reshaped = daily_vectors_tensor.numpy().reshape(user_input.shape[0], -1)
    daily_vectors_scaled = input_scaler.transform(daily_vectors_reshaped)
    daily_vectors_scaled = daily_vectors_scaled.reshape(-1, 14, 15)
    
    # Predict coal properties
    prediction_scaled = modelq.predict(daily_vectors_scaled, verbose=0)
    prediction = output_scaler.inverse_transform(prediction_scaled)
    
    # Add process parameters and predict coke properties
    conv = process_params + prediction
    conv_scaled = input__scaler.transform(conv)
    coke_prediction = rf_model.predict(conv_scaled, verbose=0)
    coke_properties = output__scaler.inverse_transform(coke_prediction)
    
    return jsonify(prediction, coke_properties),200
    


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



