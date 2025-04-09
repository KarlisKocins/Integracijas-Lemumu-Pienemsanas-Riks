from flask import Flask, render_template, request, jsonify, session
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Helper functions
def normalize_evaluations(evaluations, criteria_directions):
    """Normalize evaluation matrix"""
    eval_array = np.array(evaluations)
    norm_array = np.zeros_like(eval_array, dtype=float)
    
    for j in range(len(criteria_directions)):
        direction = criteria_directions[j]
        column = eval_array[:, j].astype(float)
        
        if direction == 1:  # Maximize
            min_val = np.min(column)
            max_val = np.max(column)
            
            if max_val == min_val:
                norm_array[:, j] = 1.0
            else:
                norm_array[:, j] = (column - min_val) / (max_val - min_val)
        else:  # Minimize
            min_val = np.min(column)
            max_val = np.max(column)
            
            if max_val == min_val:
                norm_array[:, j] = 1.0
            else:
                norm_array[:, j] = (max_val - column) / (max_val - min_val)
    
    return norm_array.tolist()

def calculate_weighted_scores(normalized_evaluations, criteria_weights):
    """Calculate weighted scores"""
    norm_array = np.array(normalized_evaluations)
    weights = np.array(criteria_weights)
    
    weighted = norm_array * weights
    return weighted.tolist()

def calculate_final_scores(weighted_scores):
    """Calculate final scores for each option"""
    weighted_array = np.array(weighted_scores)
    final_scores = np.sum(weighted_array, axis=1)
    return final_scores.tolist()

def create_results_chart(options, final_scores):
    """Create a chart of the results"""
    # Create figure with more height to accommodate labels
    plt.figure(figsize=(10, 6))
    
    # Sort options by score
    sorted_indices = np.argsort(final_scores)[::-1]
    sorted_options = [options[i] for i in sorted_indices]
    sorted_scores = [final_scores[i] * 100 for i in sorted_indices]
    
    # Create bar chart
    bars = plt.bar(sorted_options, sorted_scores, color='skyblue')
    
    # Highlight the best option
    best_idx = np.argmax(sorted_scores)
    bars[best_idx].set_color('#28a745')
    
    # Add labels and title
    plt.ylabel('Kopējais novērtējums (%)')
    plt.title('Integrācijas opciju salīdzinājums')
    
    # Add score values on top of bars
    for i, v in enumerate(sorted_scores):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Format chart
    plt.xticks(range(len(sorted_options)), sorted_options, rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(bottom=0.2)
    
    # Save to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return chart_data

# Predefined criteria sets for different organization types
PREDEFINED_CRITERIA_SETS = {
    "small_org": {
        "name": "Mazām organizācijām",
        "criteria": [
            {"name": "Izmaksas", "weight": 20, "direction": -1},
            {"name": "Ieviešanas laiks", "weight": 15, "direction": -1},
            {"name": "Piejamības zuduma risks", "weight": 30, "direction": -1},
            {"name": "Funkcionalitāte", "weight": 20, "direction": 1},
            {"name": "Darbinieku/Lietotāja gatavība", "weight": 15, "direction": 1}
        ]
    }
}

# Routes
@app.route('/')
def index():
    """Home page - reset session and start new analysis"""
    # Initialize or reset session data
    session['session_id'] = str(uuid.uuid4())
    
    # Automatically set predefined criteria for smaller organizations
    criteria_set = PREDEFINED_CRITERIA_SETS["small_org"]
    session['criteria'] = [c['name'] for c in criteria_set['criteria']]
    session['criteria_weights'] = [c['weight']/100 for c in criteria_set['criteria']]
    session['criteria_directions'] = [c['direction'] for c in criteria_set['criteria']]
    
    # Initialize other session data
    session['options'] = []
    session['option_descriptions'] = []
    session['evaluations'] = []
    
    # Suggested integration options
    suggested_options = [
        {"name": "Līdzāspastāvēšana", 
         "description": "Minimāla integrācija, sistēmas darbojas atsevišķi, dati apmainīti periodiski"},
        {"name": "Daļēja integrācija", 
         "description": "Kritisko datu sinhronizācija starp sistēmām, pārējais atstāts atsevišķi"},
        {"name": "Pilnīga integrācija", 
         "description": "Visu sistēmu apvienošana vienotā platformā"}
    ]
    
    return render_template('index.html', 
                           suggested_options=suggested_options)

@app.route('/criteria', methods=['GET', 'POST'])
def criteria():
    """Criteria definition page"""
    if request.method == 'POST':
        # Get data from form
        criteria_names = request.form.getlist('criteria_name[]')
        criteria_weights = request.form.getlist('criteria_weight[]')
        criteria_directions = request.form.getlist('criteria_direction[]')
        
        # Validate and process
        if not criteria_names or len(criteria_names) == 0:
            return render_template('criteria.html', error="Lūdzu, norādiet vismaz vienu kritēriju.")
        
        # Convert to appropriate types
        processed_weights = []
        processed_directions = []
        
        for i, name in enumerate(criteria_names):
            if not name.strip():
                continue
                
            try:
                weight = float(criteria_weights[i])
                if weight <= 0:
                    return render_template('criteria.html', 
                                          error=f"Kritērijam '{name}' svaram jābūt pozitīvam skaitlim.")
                processed_weights.append(weight)
            except (ValueError, IndexError):
                return render_template('criteria.html', 
                                      error=f"Kritērijam '{name}' svaram jābūt skaitlim.")
            
            try:
                direction = 1 if criteria_directions[i] == "1" else -1
                processed_directions.append(direction)
            except IndexError:
                processed_directions.append(1)  # Default to maximize
        
        # Normalize weights
        total_weight = sum(processed_weights)
        normalized_weights = [w/total_weight for w in processed_weights]
        
        # Save to session
        session['criteria'] = [name for name in criteria_names if name.strip()]
        session['criteria_weights'] = normalized_weights
        session['criteria_directions'] = processed_directions
        
        # Redirect to options page
        return jsonify({"redirect": "/options"})
    
    # GET request - show criteria form
    return render_template('criteria.html', 
                          criteria=session.get('criteria', []),
                          weights=session.get('criteria_weights', []),
                          directions=session.get('criteria_directions', []))

@app.route('/options', methods=['GET', 'POST'])
def options():
    """Options definition page"""
    if request.method == 'POST':
        # Get data from form
        option_names = request.form.getlist('option_name[]')
        option_descriptions = request.form.getlist('option_description[]')
        
        # Validate
        if not option_names or len(option_names) == 0:
            return render_template('options.html', error="Lūdzu, norādiet vismaz vienu integrācijas opciju.")
        
        # Process and filter empty options
        processed_names = []
        processed_descriptions = []
        
        for i, name in enumerate(option_names):
            if not name.strip():
                continue
                
            processed_names.append(name)
            try:
                processed_descriptions.append(option_descriptions[i])
            except IndexError:
                processed_descriptions.append("")
        
        # Save to session
        session['options'] = processed_names
        session['option_descriptions'] = processed_descriptions
        
        # Initialize evaluation matrix
        session['evaluations'] = [[0 for _ in range(len(session['criteria']))] for _ in range(len(processed_names))]
        
        # Redirect to evaluation page
        return jsonify({"redirect": "/evaluation"})
    
    # GET request - show options form
    return render_template('options.html', 
                          options=session.get('options', []),
                          descriptions=session.get('option_descriptions', []))

@app.route('/evaluation', methods=['GET', 'POST'])
def evaluation():
    """Evaluation page"""
    criteria = session.get('criteria', [])
    options = session.get('options', [])
    directions = session.get('criteria_directions', [])
    
    if not criteria or not options:
        return render_template('error.html', 
                              message="Kritēriji vai opcijas nav definēti. Lūdzu, atgriezieties uz sākuma lapu.")
    
    if request.method == 'POST':
        # Get evaluation data
        evaluations = []
        
        for i in range(len(options)):
            row_values = []
            for j in range(len(criteria)):
                value_key = f'evaluation_{i}_{j}'
                value_str = request.form.get(value_key, '0')
                
                try:
                    value = float(value_str)
                    row_values.append(value)
                except ValueError:
                    return render_template('evaluation.html', 
                                          criteria=criteria,
                                          options=options, 
                                          directions=directions,
                                          evaluations=session.get('evaluations', []),
                                          error=f"Opcijas '{options[i]}' vērtībai pēc kritērija '{criteria[j]}' jābūt skaitlim.")
            
            evaluations.append(row_values)
        
        # Save to session
        session['evaluations'] = evaluations
        
        # Redirect to results page
        return jsonify({"redirect": "/results"})
    
    # GET request - show evaluation matrix
    return render_template('evaluation.html', 
                          criteria=criteria,
                          options=options,
                          directions=directions,
                          evaluations=session.get('evaluations', []))

@app.route('/results')
def results():
    """Results page"""
    # Get data from session
    criteria = session.get('criteria', [])
    criteria_weights = session.get('criteria_weights', [])
    criteria_directions = session.get('criteria_directions', [])
    options = session.get('options', [])
    option_descriptions = session.get('option_descriptions', [])
    evaluations = session.get('evaluations', [])
    
    if not criteria or not options or not evaluations:
        return render_template('error.html', 
                              message="Nav pietiekami datu rezultātu aprēķināšanai. Lūdzu, atgriezieties uz sākuma lapu.")
    
    # Perform calculations
    normalized = normalize_evaluations(evaluations, criteria_directions)
    weighted = calculate_weighted_scores(normalized, criteria_weights)
    final_scores = calculate_final_scores(weighted)
    
    # Create chart
    chart_data = create_results_chart(options, final_scores)
    
    # Find the best option
    best_option_index = np.argmax(final_scores)
    best_option = options[best_option_index]
    best_score = final_scores[best_option_index]
    
    # Calculate percentages
    score_percentages = [score * 100 for score in final_scores]
    
    # Find strengths and weaknesses
    normalized_scores = np.array(normalized[best_option_index])
    strengths = []
    weaknesses = []
    
    for i, score in enumerate(normalized_scores):
        if score > 0.7:
            strengths.append(criteria[i])
        elif score < 0.3:
            weaknesses.append(criteria[i])
    
    # Prepare ranking
    ranked_indices = np.argsort(final_scores)[::-1]
    ranking = []
    
    for i, idx in enumerate(ranked_indices):
        ranking.append({
            "rank": i+1,
            "name": options[idx],
            "score": final_scores[idx],
            "percentage": score_percentages[idx]
        })
    
    # Compare to second best if it exists
    comparison = None
    if len(ranked_indices) > 1:
        second_best_idx = ranked_indices[1]
        second_best = options[second_best_idx]
        diff = score_percentages[best_option_index] - score_percentages[second_best_idx]
        
        if diff < 5:
            comparison = {
                "name": second_best,
                "diff": diff,
                "close": True
            }
        else:
            comparison = {
                "name": second_best,
                "diff": diff,
                "close": False
            }
    
    # Render results template
    return render_template('results.html',
                          best_option=best_option,
                          best_score=best_score,
                          best_percentage=score_percentages[best_option_index],
                          ranking=ranking,
                          strengths=strengths,
                          weaknesses=weaknesses,
                          comparison=comparison,
                          description=option_descriptions[best_option_index],
                          chart_data=chart_data)

@app.route('/reset')
def reset():
    """Reset session and start over"""
    session.clear()
    return jsonify({"redirect": "/"})

if __name__ == '__main__':
    app.run(debug=True) 