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
    try:
        # Parse request data
        data = request.json  
        if not data:
            raise ValueError("No data received in the request")
        
        # Get model parameters from request
        coal_blends = data.get("blends", [])
        coal_types = [blk["coalType"] for blk in coal_blends]
        min_percentages = np.array([int(blk["minPercentage"]) for blk in coal_blends])
        max_percentages = np.array([int(blk["maxPercentage"]) for blk in coal_blends])
        
        # Pad to fixed length
        min_percentages_padded = np.pad(min_percentages, (0, 14 - len(min_percentages)), mode='constant')
        max_percentages_padded = np.pad(max_percentages, (0, 14 - len(max_percentages)), mode='constant')
        
        # Get desired parameters
        desired_coke_params = data.get("cokeParameters", {})
        Option = int(data.get("processType", 1))  # Default to 1 if not provided
        proces_para = data.get("processParameters", {})
        
        # Handle user input for blend
        oneblends = data.get('blendcoal', [])
        user_input_values_padded = np.zeros((1, 14))
        
        if oneblends:
            user_input_values = np.array([blend['currentRange'] for blend in oneblends])
            if user_input_values.sum() != 100:
                return jsonify({"error": "The total of current range must add up to 100."}), 400
            user_input_values_padded[0, :len(user_input_values)] = user_input_values
        
        # Load models and data - use cached versions if available
        models, scalers, data_matrices = load_models_and_data(Option)
        modelq, rf_model = models
        input_scaler, output_scaler, input__scaler, output__scaler = scalers
        D, P, coal_data = data_matrices
        
        # Process the coal cost data
        coal_costs = get_coal_costs(coal_types, coal_data)
        
        # Get min/max values for validation
        min_max_values = read_min_max_values()
        
        # Generate combinations in batches to save memory
        valid_results = []
        batch_size = 1000  # Adjust based on memory constraints
        
        for combo_batch in batch_generator(min_percentages_padded, max_percentages_padded, batch_size):
            # Process this batch of combinations
            predictions_batch, properties_batch = process_combination_batch(
                combo_batch, P, modelq, rf_model, 
                input_scaler, output_scaler, input__scaler, output__scaler,
                proces_para, Option
            )
            
            # Filter valid combinations
            valid_indices = filter_valid_predictions(
                predictions_batch, min_max_values, 
                desired_coke_params["ASH"], desired_coke_params["VM"], 
                desired_coke_params["M_40MM"], desired_coke_params["M_10MM"],
                desired_coke_params["CSR"], desired_coke_params["CRI"], 
                desired_coke_params["AMS"]
            )
            
            # Save valid results
            for idx in valid_indices:
                valid_results.append({
                    'combination': combo_batch[idx],
                    'prediction': predictions_batch[idx],
                    'properties': properties_batch[idx],
                    'cost': calculate_cost(combo_batch[idx], coal_costs)
                })
                
            # Free memory
            del predictions_batch, properties_batch
            gc.collect()
            
        # No valid combinations found
        if not valid_results:
            return jsonify({"error": "No valid combinations found with given constraints"}), 400
        
        # Calculate scores and rank results
        performance_scores = [calculate_performance_score(r['prediction'], desired_coke_params, min_max_values) 
                             for r in valid_results]
        cost_scores = [r['cost'] for r in valid_results]
        
        # Normalize scores
        norm_perf = normalize_array(performance_scores)
        norm_cost = normalize_array(cost_scores)
        
        # Calculate combined scores with weights
        cost_weight = min_max_values['cost_weightage']
        performance_weight = min_max_values['coke_quality']
        combined_scores = [(cost_weight * nc) + (performance_weight * np) 
                          for nc, np in zip(norm_cost, norm_perf)]
        
        # Get the three representative blends
        # Best performance
        best_perf_idx = np.argmin(performance_scores)
        # Lowest cost
        best_cost_idx = np.argmin(cost_scores)
        # Best combined
        best_combined_idx = np.argmin(combined_scores)
        
        # Process user's proposed blend if provided
        proposed_result = None
        if np.any(user_input_values_padded != 0):
            proposed_result = predict_for_blend(
                user_input_values_padded, P, modelq, rf_model,
                input_scaler, output_scaler, input__scaler, output__scaler,
                proces_para
            )
        
        # Prepare response
        response = {
            "blend1": format_blend_result(valid_results[best_perf_idx]),
            "blend2": format_blend_result(valid_results[best_cost_idx]),
            "blend3": format_blend_result(valid_results[best_combined_idx]),
            "valid_predictions_count": len(valid_results)
        }
        
        if proposed_result:
            response["ProposedCoal"] = {
                "Blend2": proposed_result['properties'].tolist(),
                "Coke2": proposed_result['prediction'].tolist()
            }
        else:
            response["ProposedCoal"] = {
                "error": "No valid blendcoal data provided, unable to make predictions."
            }
            
        # Clean up memory before returning
        clean_memory()
            
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Helper functions for the main route
def load_models_and_data(Option):
    # Global cache for models and data
    global _models_cache, _data_cache
    
    # Initialize caches if not exists
    if not hasattr(load_models_and_data, '_models_cache'):
        load_models_and_data._models_cache = {}
    if not hasattr(load_models_and_data, '_data_cache'):
        load_models_and_data._data_cache = {}
    
    # Load or retrieve models
    if f'models_{Option}' not in load_models_and_data._models_cache:
        # Load models and scalers based on Option
        modelq = load_model_from_disk('modelq.h5')  # Load saved model
        rf_model = load_model_from_disk('rf_model.h5')  # Load saved model
        
        # Load scalers
        input_scaler = load_scaler('input_scaler.pkl')
        output_scaler = load_scaler('output_scaler.pkl')
        input__scaler = load_scaler('input__scaler.pkl')
        output__scaler = load_scaler('output__scaler.pkl')
        
        # Cache the models and scalers
        load_models_and_data._models_cache[f'models_{Option}'] = (modelq, rf_model)
        load_models_and_data._models_cache[f'scalers_{Option}'] = (
            input_scaler, output_scaler, input__scaler, output__scaler
        )
    
    # Load or retrieve data matrices
    if 'data_matrices' not in load_models_and_data._data_cache:
        D = np.loadtxt('coal_percentages.csv', delimiter=',')
        P = np.loadtxt('Individual_coal_properties.csv', delimiter=',')
        coal_data = pd.read_csv('individual_coal_prop.csv', dtype=str, header=None, on_bad_lines='skip')
        
        load_models_and_data._data_cache['data_matrices'] = (D, P, coal_data)
    
    return (
        load_models_and_data._models_cache[f'models_{Option}'],
        load_models_and_data._models_cache[f'scalers_{Option}'],
        load_models_and_data._data_cache['data_matrices']
    )

def batch_generator(min_percentages, max_percentages, batch_size=1000):
    """Generate combinations in batches to save memory"""
    total_batches = 0
    current_batch = []
    
    def generate_combinations_helper(index, current_combination, current_sum):
        nonlocal current_batch, total_batches
        
        if len(current_batch) >= batch_size:
            yield np.array(current_batch)
            current_batch = []
            
        if index == len(min_percentages) - 1:
            remaining = 100 - current_sum
            if min_percentages[index] <= remaining <= max_percentages[index]:
                current_batch.append(current_combination + [remaining])
                total_batches += 1
            return
            
        for value in range(int(min_percentages[index]), int(max_percentages[index]) + 1):
            if current_sum + value <= 100:
                yield from generate_combinations_helper(
                    index + 1, current_combination + [value], current_sum + value
                )
    
    yield from generate_combinations_helper(0, [], 0)
    
    # Yield any remaining combinations
    if current_batch:
        yield np.array(current_batch)

def process_combination_batch(combinations, P, modelq, rf_model, 
                             input_scaler, output_scaler, input__scaler, output__scaler, 
                             proces_para, Option):
    """Process a batch of combinations"""
    # Convert to tensor
    D_tensor = tf.constant(combinations, dtype=tf.float32)
    P_tensor = tf.constant(P, dtype=tf.float32)
    
    # Create daily vectors more efficiently with tensor operations
    daily_vectors = []
    for i in range(D_tensor.shape[0]):
        # Use broadcasting for more efficient computation
        product_vectors = tf.expand_dims(D_tensor[i], 1) * P_tensor
        daily_vectors.append(product_vectors)
    
    daily_vectors_tensor = tf.stack(daily_vectors)
    
    # Flatten for scaling
    batch_size = daily_vectors_tensor.shape[0]
    daily_vectors_flattened = tf.reshape(daily_vectors_tensor, [batch_size, -1])
    
    # Scale inputs
    daily_vectors_scaled = input_scaler.transform(daily_vectors_flattened.numpy())
    daily_vectors_scaled = daily_vectors_scaled.reshape(-1, 14, 15)
    
    # Predict in smaller sub-batches to reduce memory usage
    sub_batch_size = 64  # Adjust based on available memory
    blend_predictions = []
    
    for i in range(0, len(daily_vectors_scaled), sub_batch_size):
        sub_batch = daily_vectors_scaled[i:i+sub_batch_size]
        sub_predictions = modelq.predict(sub_batch, verbose=0)
        blend_predictions.append(sub_predictions)
    
    blend_predictions = np.vstack(blend_predictions)
    blended_coal_properties = output_scaler.inverse_transform(blend_predictions)
    
    # Process for second model
    if Option == 3:
        proces_para = np.pad(proces_para, (0, 2), mode='constant', constant_values=0)
        
    # Add process parameters to each prediction
    blend_with_process = blend_predictions + proces_para
    
    # Scale for second model
    blend_scaled = input__scaler.transform(blend_with_process)
    
    # Predict coke properties in sub-batches
    coke_predictions = []
    for i in range(0, len(blend_scaled), sub_batch_size):
        sub_batch = blend_scaled[i:i+sub_batch_size]
        sub_predictions = rf_model.predict(sub_batch, verbose=0)
        coke_predictions.append(sub_predictions)
    
    coke_predictions = np.vstack(coke_predictions)
    final_predictions = output__scaler.inverse_transform(coke_predictions)
    
    return final_predictions, blended_coal_properties

def filter_valid_predictions(predictions, min_max_values, desired_ash, desired_vm, 
                           desired_m40, desired_m10, desired_csr, desired_cri, desired_ams):
    """Filter predictions to only include valid ones"""
    valid_indices = []
    
    for i, prediction in enumerate(predictions):
        if (min_max_values['ash']['lower'] <= prediction[0] <= min_max_values['ash']['upper'] and
            min_max_values['vm']['lower'] <= prediction[1] <= min_max_values['vm']['upper'] and
            min_max_values['m40']['lower'] <= prediction[9] <= min_max_values['m40']['upper'] and
            min_max_values['m10']['lower'] <= prediction[10] <= min_max_values['m10']['upper'] and
            min_max_values['csr']['lower'] <= prediction[12] <= min_max_values['csr']['upper'] and
            min_max_values['cri']['lower'] <= prediction[13] <= min_max_values['cri']['upper'] and
            min_max_values['ams']['lower'] <= prediction[14] <= min_max_values['ams']['upper']):
            valid_indices.append(i)
    
    return valid_indices

def calculate_performance_score(prediction, desired_params, min_max_values):
    """Calculate performance score based on desired parameters"""
    diff_ash = ((desired_params["ASH"] - prediction[0]) / desired_params["ASH"]) * min_max_values['ash']['weight']
    diff_vm = ((desired_params["VM"] - prediction[1]) / desired_params["VM"]) * min_max_values['vm']['weight']
    diff_m40 = ((prediction[9] - desired_params["M_40MM"]) / desired_params["M_40MM"]) * min_max_values['m40']['weight']
    diff_m10 = ((desired_params["M_10MM"] - prediction[10]) / desired_params["M_10MM"]) * min_max_values['m10']['weight']
    diff_csr = ((prediction[12] - desired_params["CSR"]) / desired_params["CSR"]) * min_max_values['csr']['weight']
    diff_cri = ((desired_params["CRI"] - prediction[13]) / desired_params["CRI"]) * min_max_values['cri']['weight']
    diff_ams = ((prediction[14] - desired_params["AMS"]) / desired_params["AMS"]) * min_max_values['ams']['weight']
    
    return sum([diff_ash, diff_vm, diff_m40, diff_m10, diff_csr, diff_cri, diff_ams])

def normalize_array(arr):
    """Normalize array to 0-1 range"""
    min_val = min(arr)
    max_val = max(arr)
    if max_val == min_val:
        return [0] * len(arr)
    return [(x - min_val) / (max_val - min_val) for x in arr]

def calculate_cost(blend, coal_costs):
    """Calculate cost of a blend"""
    return sum(blend[i] * coal_costs[i] / 100 for i in range(min(len(blend), len(coal_costs))))

def format_blend_result(result):
    """Format a blend result for the response"""
    return {
        "composition": result['combination'].tolist(),
        "blendedcoal": result['properties'].tolist(),
        "properties": result['prediction'].tolist(),
        "cost": result['cost']
    }

def clean_memory():
    """Clean up memory after processing"""
    gc.collect()
    tf.keras.backend.clear_session()

def load_model_from_disk(model_path):
    """Load model from disk with error handling"""
    try:
        return keras.models.load_model(model_path)
    except:
        # Log error and fallback - in production you might want to raise
        print(f"Error loading model from {model_path}")
        return None

def load_scaler(scaler_path):
    """Load scaler from disk with error handling"""
    try:
        import pickle
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    except:
        print(f"Error loading scaler from {scaler_path}")
        return None

def predict_for_blend(blend, P, modelq, rf_model, input_scaler, output_scaler, input__scaler, output__scaler, proces_para):
    """Make predictions for a user-provided blend"""
    D_tensor = tf.constant(blend, dtype=tf.float32)
    P_tensor = tf.constant(P, dtype=tf.float32)
    
    # Create daily vector for the blend
    product_vectors = tf.expand_dims(D_tensor[0], 1) * P_tensor
    daily_vectors_tensor = tf.expand_dims(product_vectors, 0)
    
    # Flatten and scale
    daily_vectors_flattened = tf.reshape(daily_vectors_tensor, [1, -1])
    daily_vectors_scaled = input_scaler.transform(daily_vectors_flattened.numpy())
    daily_vectors_scaled = daily_vectors_scaled.reshape(-1, 14, 15)
    
    # Predict blended coal properties
    blended_coal_pred = modelq.predict(daily_vectors_scaled, verbose=0)
    blended_coal = output_scaler.inverse_transform(blended_coal_pred)
    
    # Add process parameters
    conv = blended_coal_pred + proces_para
    
    # Predict coke properties
    conv_scaled = input__scaler.transform(conv)
    coke_pred = rf_model.predict(conv_scaled, verbose=0)
    coke_properties = output__scaler.inverse_transform(coke_pred)
    
    return {
        'properties': blended_coal[0],
        'prediction': coke_properties[0]
    }

def read_min_max_values():
    """Read min-max values from CSV file with caching"""
    if not hasattr(read_min_max_values, 'cache'):
        df = pd.read_csv('min-maxvalues.csv')
        read_min_max_values.cache = {
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
    return read_min_max_values.cache

def get_coal_costs(coal_types, coal_data):
    """Get coal costs from coal data"""
    coal_costs = []
    for coal_type in coal_types:
        try:
            cost = float(coal_data.loc[coal_data[0] == coal_type, coal_data.columns[-2]].values[0])
            coal_costs.append(cost)
        except (IndexError, ValueError):
            coal_costs.append(0)  # Default value if cost not found
    return coal_costs
    


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



