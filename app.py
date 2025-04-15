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
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import itertools
from functools import lru_cache

app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})

# Pre-loaded data and models
DATA = {
    'P': None,          # Individual_coal_properties
    'D': None,          # coal_percentages
    'Blended_coal': None,
    'Coke_properties': None,
    'coal_costs': None,
    'min_max': None,
    'process_params': None
}

MODELS = {
    'modelq': None,
    'rf_model': None
}

SCALERS = {
    'input': None,
    'output': None,
    'input_phase2': None,
    'output_phase2': None
}

def initialize_app():
    """Initialize all required data, models, and scalers"""
    print("Initializing application...")
    
    try:
        # 1. Load all data files
        print("Loading data files...")
        data_files = {
            'P': 'Individual_coal_properties.csv',
            'D': 'coal_percentages.csv',
            'Blended_coal': 'blended_coal_data.csv',
            'Coke_properties': 'coke_properties.csv'
        }
        
        for key, filename in data_files.items():
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Data file {filename} not found")
            DATA[key] = np.loadtxt(filename, delimiter=',')
            print(f"Loaded {filename} successfully")
        
        # Load coal costs
        print("Loading coal costs...")
        if not os.path.exists('individual_coal_prop.csv'):
            raise FileNotFoundError("individual_coal_prop.csv not found")
        coal_data = pd.read_csv('individual_coal_prop.csv', dtype=str, header=None, on_bad_lines='skip')
        DATA['coal_costs'] = {row[0]: float(row[-2]) for _, row in coal_data.iterrows()}
        print("Loaded coal costs successfully")
        
        # Load min-max values
        print("Loading min-max values...")
        if not os.path.exists('min-maxvalues.csv'):
            raise FileNotFoundError("min-maxvalues.csv not found")
        df = pd.read_csv('min-maxvalues.csv')
        DATA['min_max'] = {
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
        print("Loaded min-max values successfully")
        
        # Load process parameters
        print("Loading process parameters...")
        process_files = {
            1: 'Process_parameter_for_Rec_Top_Char.csv',
            2: 'Process_parameter_for_Rec_Stam_Char.csv',
            3: 'Process_parameter_for_Non_Rec_Stam_Char.csv'
        }
        DATA['process_params'] = {}
        for opt, filename in process_files.items():
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Process file {filename} not found")
            DATA['process_params'][opt] = np.loadtxt(filename, delimiter=',')
            if opt == 3:  # Special padding for option 3
                DATA['process_params'][opt] = np.pad(DATA['process_params'][opt], ((0, 0), (0, 2)), mode='constant')
            print(f"Loaded {filename} successfully")
        
        # 2. Initialize or load models and scalers
        print("Loading models and scalers...")
        model_files = ['modelq.h5', 'rf_model.h5']
        scaler_files = ['input_scaler.pkl', 'output_scaler.pkl', 'input_phase2_scaler.pkl', 'output_phase2_scaler.pkl']
        
        if all(os.path.exists(f) for f in model_files + scaler_files):
            # Load pre-trained models and scalers
            MODELS['modelq'] = tf.keras.models.load_model('modelq.h5')
            MODELS['rf_model'] = tf.keras.models.load_model('rf_model.h5')
            SCALERS['input'] = joblib.load('input_scaler.pkl')
            SCALERS['output'] = joblib.load('output_scaler.pkl')
            SCALERS['input_phase2'] = joblib.load('input_phase2_scaler.pkl')
            SCALERS['output_phase2'] = joblib.load('output_phase2_scaler.pkl')
            print("Loaded pre-trained models and scalers")
        else:
            print("Required files missing, training new models...")
            train_models()
            
        print("Initialization complete")
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise

def train_models():
    """Train and save all models and scalers"""
    print("Training models...")
    try:
        # Model 1 - modelq
        print("Training modelq...")
        daily_vectors = DATA['D'][:, :, None] * DATA['P'][None, :, :]
        X_train, X_test, y_train, y_test = train_test_split(
            daily_vectors, DATA['Blended_coal'], test_size=0.2, random_state=42
        )
        
        SCALERS['input'] = MinMaxScaler()
        SCALERS['output'] = MinMaxScaler()
        
        X_train_scaled = SCALERS['input'].fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_test_scaled = SCALERS['input'].transform(X_test.reshape(X_test.shape[0], -1))
        y_train_scaled = SCALERS['output'].fit_transform(y_train)
        y_test_scaled = SCALERS['output'].transform(y_test)
        
        MODELS['modelq'] = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(14*15,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(15, activation='linear')
        ])
        
        MODELS['modelq'].compile(optimizer='adam', loss='mse')
        MODELS['modelq'].fit(
            X_train_scaled, y_train_scaled,
            epochs=100, batch_size=8,
            validation_data=(X_test_scaled, y_test_scaled),
            verbose=0
        )
        
        # Model 2 - rf_model
        print("Training rf_model...")
        Conv_matrix = DATA['Blended_coal'] + DATA['process_params'][1]
        X_train, X_test, y_train, y_test = train_test_split(
            Conv_matrix, DATA['Coke_properties'], test_size=0.2, random_state=42
        )
        
        SCALERS['input_phase2'] = MinMaxScaler()
        SCALERS['output_phase2'] = MinMaxScaler()
        
        X_train_scaled = SCALERS['input_phase2'].fit_transform(X_train)
        X_test_scaled = SCALERS['input_phase2'].transform(X_test)
        y_train_scaled = SCALERS['output_phase2'].fit_transform(y_train)
        y_test_scaled = SCALERS['output_phase2'].transform(y_test)
        
        MODELS['rf_model'] = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(15,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(15, activation='linear')
        ])
        
        MODELS['rf_model'].compile(optimizer='adam', loss='mse')
        MODELS['rf_model'].fit(
            X_train_scaled, y_train_scaled,
            epochs=100, batch_size=8,
            validation_data=(X_test_scaled, y_test_scaled),
            verbose=0
        )
        
        # Save everything
        print("Saving models and scalers...")
        MODELS['modelq'].save('modelq.h5')
        MODELS['rf_model'].save('rf_model.h5')
        joblib.dump(SCALERS['input'], 'input_scaler.pkl')
        joblib.dump(SCALERS['output'], 'output_scaler.pkl')
        joblib.dump(SCALERS['input_phase2'], 'input_phase2_scaler.pkl')
        joblib.dump(SCALERS['output_phase2'], 'output_phase2_scaler.pkl')
        
        print("Models trained and saved successfully")
        
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        raise

@lru_cache(maxsize=100)
def generate_combinations(min_percentages, max_percentages):
    """Generate valid combinations with caching"""
    ranges = [range(max(0, min_p), min(100, max_p)+1) 
              for min_p, max_p in zip(min_percentages, max_percentages)]
    valid = [combo for combo in itertools.product(*ranges) if sum(combo) == 100]
    return np.array(valid)

def process_user_blend(user_input, process_params):
    """Process the user-provided blend"""
    if not user_input:
        return None
    
    try:
        user_values = np.array([blend['currentRange'] for blend in user_input])
        if user_values.sum() != 100:
            return {"error": "The total of current range must add up to 100."}
        
        user_values_padded = np.pad(user_values, (0, 14 - len(user_values)), 'constant')
        
        # Vectorized calculation
        daily_vector = user_values_padded[:, None] * DATA['P']
        daily_vector_scaled = SCALERS['input'].transform(daily_vector.reshape(1, -1))
        
        # Predict blended coal properties
        blended_coal = MODELS['modelq'].predict(daily_vector_scaled)
        blended_coal = SCALERS['output'].inverse_transform(blended_coal)
        
        # Predict coke properties
        conv_matrix = blended_coal + process_params
        conv_matrix_scaled = SCALERS['input_phase2'].transform(conv_matrix)
        coke_properties = MODELS['rf_model'].predict(conv_matrix_scaled)
        coke_properties = SCALERS['output_phase2'].inverse_transform(coke_properties)
        
        return {
            "Blend2": blended_coal[0].tolist(),
            "Coke2": coke_properties[0].tolist()
        }
    except Exception as e:
        return {"error": f"Error processing user blend: {str(e)}"}
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
    try:
        # Verify all components are loaded
        if None in SCALERS.values() or None in MODELS.values():
            initialize_app()
            if None in SCALERS.values() or None in MODELS.values():
                return jsonify({"error": "System not properly initialized"}), 500
                
        data = request.json  
        if not data:
            return jsonify({"error": "No data received"}), 400
            
        # Extract and validate input data
        coal_blends = data.get("blends", [])
        if not coal_blends:
            return jsonify({"error": "No coal blends provided"}), 400
            
        coal_types = [blk.get("coalType") for blk in coal_blends]
        if None in coal_types:
            return jsonify({"error": "Missing coalType in blends"}), 400
            
        min_percentages = []
        max_percentages = []
        for blk in coal_blends:
            try:
                min_percentages.append(int(blk.get("minPercentage", 0)))
                max_percentages.append(int(blk.get("maxPercentage", 100)))
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid minPercentage or maxPercentage"}), 400
        
        desired_coke_params = data.get("cokeParameters", {})
        oneblends = data.get('blendcoal', [])
        
        try:
            Option = int(data.get("processType", 1))
            if Option not in [1, 2, 3]:
                return jsonify({"error": "processType must be 1, 2, or 3"}), 400
        except ValueError:
            return jsonify({"error": "Invalid processType"}), 400
            
        # Process parameters handling
        proces_para = data.get("processParameters", {})
        if isinstance(proces_para, dict):
            proces_para = list(proces_para.values())
        try:
            proces_para = np.array(proces_para, dtype=float)
        except ValueError:
            return jsonify({"error": "Invalid processParameters values"}), 400
        
        # Pad arrays
        min_percentages_padded = np.pad(min_percentages, (0, 14 - len(min_percentages)), 'constant') 
        max_percentages_padded = np.pad(max_percentages, (0, 14 - len(max_percentages)), 'constant')
        
        if Option == 3:
            proces_para = np.pad(proces_para, (0, 2), 'constant')
        
        # Generate all valid combinations
        all_combinations = generate_combinations(
            tuple(min_percentages_padded),
            tuple(max_percentages_padded)
        )
        
        # Process all combinations
        results = []
        for comb in all_combinations:
            try:
                # Vectorized calculation
                daily_vector = comb[:, None] * DATA['P']
                daily_vector_scaled = SCALERS['input'].transform(daily_vector.reshape(1, -1))
                
                # Predict blended coal properties
                blended_coal = MODELS['modelq'].predict(daily_vector_scaled)
                blended_coal = SCALERS['output'].inverse_transform(blended_coal)
                
                # Predict coke properties
                conv_matrix = blended_coal + proces_para
                conv_matrix_scaled = SCALERS['input_phase2'].transform(conv_matrix)
                coke_properties = MODELS['rf_model'].predict(conv_matrix_scaled)
                coke_properties = SCALERS['output_phase2'].inverse_transform(coke_properties)
                
                # Calculate cost
                cost = sum(comb[i] * DATA['coal_costs'][coal_types[i]] / 100 
                         for i in range(min(len(comb), len(coal_types))))
                
                results.append({
                    'combination': comb,
                    'blended_coal': blended_coal[0],
                    'coke_properties': coke_properties[0],
                    'cost': cost
                })
            except Exception as e:
                print(f"Error processing combination: {str(e)}")
                continue
        
        # Filter valid predictions
        min_max = DATA['min_max']
        valid_results = [
            r for r in results if (
                min_max['ash']['lower'] <= r['coke_properties'][0] <= min_max['ash']['upper'] and
                min_max['vm']['lower'] <= r['coke_properties'][1] <= min_max['vm']['upper'] and
                min_max['m40']['lower'] <= r['coke_properties'][9] <= min_max['m40']['upper'] and
                min_max['m10']['lower'] <= r['coke_properties'][10] <= min_max['m10']['upper'] and
                min_max['csr']['lower'] <= r['coke_properties'][12] <= min_max['csr']['upper'] and
                min_max['cri']['lower'] <= r['coke_properties'][13] <= min_max['cri']['upper'] and
                min_max['ams']['lower'] <= r['coke_properties'][14] <= min_max['ams']['upper']
            )
        ]
        
        if not valid_results:
            return jsonify({"error": "No valid combinations found"}), 400
        
        # Calculate quality differences
        desired = {
            'ash': desired_coke_params.get("ASH", 0),
            'vm': desired_coke_params.get("VM", 0),
            'm40': desired_coke_params.get("M_40MM", 0),
            'm10': desired_coke_params.get("M_10MM", 0),
            'csr': desired_coke_params.get("CSR", 0),
            'cri': desired_coke_params.get("CRI", 0),
            'ams': desired_coke_params.get("AMS", 0)
        }
        
        for result in valid_results:
            props = result['coke_properties']
            diff = [
                ((desired['ash'] - props[0]) / max(desired['ash'], 1e-6)) * min_max['ash']['weight'],
                ((desired['vm'] - props[1]) / max(desired['vm'], 1e-6)) * min_max['vm']['weight'],
                ((props[9] - desired['m40']) / max(desired['m40'], 1e-6)) * min_max['m40']['weight'],
                ((desired['m10'] - props[10]) / max(desired['m10'], 1e-6)) * min_max['m10']['weight'],
                ((props[12] - desired['csr']) / max(desired['csr'], 1e-6)) * min_max['csr']['weight'],
                ((desired['cri'] - props[13]) / max(desired['cri'], 1e-6)) * min_max['cri']['weight'],
                ((props[14] - desired['ams']) / max(desired['ams'], 1e-6)) * min_max['ams']['weight']
            ]
            result['total_diff'] = sum(diff)
            result['diffs'] = diff
        
        # Sort by quality
        valid_results.sort(key=lambda x: x['total_diff'], reverse=True)
        
        # Sort by cost
        sorted_by_cost = sorted(valid_results, key=lambda x: x['cost'])
        
        # Combine cost and quality
        costs = np.array([r['cost'] for r in valid_results])
        quals = np.array([r['total_diff'] for r in valid_results])
        norm_costs = (costs - costs.min()) / max((costs.max() - costs.min()), 1e-6)
        norm_quals = (quals - quals.min()) / max((quals.max() - quals.min()), 1e-6)
        combined_scores = (min_max['cost_weightage'] * norm_costs + 
                         min_max['coke_quality'] * norm_quals)
        best_combined_idx = np.argmin(combined_scores)
        
        # Prepare the three recommended blends
        blend_1 = valid_results[0]
        blend_2 = sorted_by_cost[0]
        blend_3 = valid_results[best_combined_idx]
        
        # Process user blend if provided
        proposed_coal = process_user_blend(oneblends, proces_para)
        if isinstance(proposed_coal, dict) and 'error' in proposed_coal:
            return jsonify(proposed_coal), 400
        
        # Prepare final response
        response = {
            "blend1": {
                "composition": blend_1['combination'].tolist(),
                "blendedcoal": blend_1['blended_coal'].tolist(),
                "properties": blend_1['coke_properties'].tolist(),
                "cost": float(blend_1['cost'])
            },
            "blend2": {
                "composition": blend_2['combination'].tolist(),
                "blendedcoal": blend_2['blended_coal'].tolist(),
                "properties": blend_2['coke_properties'].tolist(),
                "cost": float(blend_2['cost'])
            },
            "blend3": {
                "composition": blend_3['combination'].tolist(),
                "blendedcoal": blend_3['blended_coal'].tolist(),
                "properties": blend_3['coke_properties'].tolist(),
                "cost": float(blend_3['cost'])
            },
            "valid_predictions_count": len(valid_results),
            "ProposedCoal": proposed_coal or {
                "error": "No valid blendcoal data provided, unable to make predictions."
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in cost endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500
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

@app.before_first_request
def startup():
    """Initialize before first request"""
    try:
        initialize_app()
    except Exception as e:
        print(f"Failed to initialize: {str(e)}")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



