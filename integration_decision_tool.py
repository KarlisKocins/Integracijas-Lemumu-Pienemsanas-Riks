import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Predefined criteria for smaller organizations
PREDEFINED_CRITERIA = [
    {"name": "Izmaksas", "weight": 20, "direction": "Minimizēt"},
    {"name": "Ieviešanas laiks", "weight": 15, "direction": "Minimizēt"},
    {"name": "Piejamības zuduma risks", "weight": 30, "direction": "Minimizēt"},
    {"name": "Funkcionalitāte", "weight": 20, "direction": "Maksimizēt"},
    {"name": "Darbinieku/Lietotāja gatavība", "weight": 15, "direction": "Maksimizēt"}
]

class IntegrationDecisionTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrācijas lēmumu pieņemšanas rīks")
        self.root.geometry("1000x700")
        
        # Data structures
        self.criteria = []  # List of criteria names
        self.criteria_weights = []  # List of weights for each criterion
        self.options = []  # List of integration option names
        self.option_descriptions = []  # List of descriptions for each option
        self.evaluations = []  # 2D matrix of evaluations [option][criterion]
        self.normalized_evaluations = []  # 2D matrix of normalized evaluations
        self.weighted_scores = []  # 2D matrix of weighted scores
        self.final_scores = []  # List of final scores for each option
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs for each module
        self.criteria_tab = ttk.Frame(self.notebook)
        self.options_tab = ttk.Frame(self.notebook)
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.criteria_tab, text="1. Kritēriju definēšana")
        self.notebook.add(self.options_tab, text="2. Integrācijas opcijas")
        self.notebook.add(self.evaluation_tab, text="3. Novērtēšana")
        self.notebook.add(self.results_tab, text="4. Lēmumu atbalsts")
        
        # Setup each tab
        self.setup_criteria_tab()
        self.setup_options_tab()
        self.setup_evaluation_tab()
        self.setup_results_tab()
        
        # Bind tab change event to update data
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def setup_criteria_tab(self):
        """Setup the criteria definition tab"""
        frame = ttk.Frame(self.criteria_tab, padding="10 10 10 10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Kritēriju definēšana", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=10, sticky='w')
        
        info_text = ("Definējiet novērtēšanas kritērijus un to svarus. "
                    "Svariem jābūt pozitīviem skaitļiem, kas kopā veido 100%.")
        ttk.Label(frame, text=info_text, wraplength=800).grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky='w')
        
        # Info about predefined criteria
        info_label = ttk.Label(frame, text="Zemāk ir iepriekš definēta kritēriju kopa, kas pielāgota mazām organizācijām. Jūs varat to rediģēt pēc vajadzības.", 
                         wraplength=800, foreground="blue")
        info_label.grid(row=2, column=0, columnspan=3, pady=(0, 10), sticky='w')
        
        # Recommended criteria categories
        ttk.Label(frame, text="Ieteicamās kritēriju kategorijas:", font=("Arial", 10, "bold")).grid(row=3, column=0, columnspan=3, pady=(10, 5), sticky='w')
        categories = ["Finanšu (piem., izmaksas, ROI)", 
                      "Tehniskie (piem., saderība, drošība)", 
                      "Organizatoriskie (piem., personāla gatavība)", 
                      "Riski (piem., ieviešanas risks)"]
        
        for i, category in enumerate(categories):
            ttk.Label(frame, text=f"• {category}").grid(row=4+i, column=0, columnspan=3, sticky='w')
        
        # Criteria input section
        ttk.Label(frame, text="Kritērija nosaukums", font=("Arial", 10, "bold")).grid(row=9, column=0, pady=(20, 5), sticky='w')
        ttk.Label(frame, text="Svars (%)", font=("Arial", 10, "bold")).grid(row=9, column=1, pady=(20, 5), sticky='w')
        ttk.Label(frame, text="Mērķis", font=("Arial", 10, "bold")).grid(row=9, column=2, pady=(20, 5), sticky='w')
        
        # Frame for criteria entries (scrollable)
        self.criteria_entries_frame = ttk.Frame(frame)
        self.criteria_entries_frame.grid(row=10, column=0, columnspan=3, sticky='nsew')
        
        # Scrollable canvas for criteria entries
        canvas = tk.Canvas(self.criteria_entries_frame)
        scrollbar = ttk.Scrollbar(self.criteria_entries_frame, orient="vertical", command=canvas.yview)
        self.scrollable_criteria_frame = ttk.Frame(canvas)
        
        self.scrollable_criteria_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_criteria_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initial criteria rows
        self.criteria_rows = []
        
        # Use predefined criteria for smaller organizations
        for criteria in PREDEFINED_CRITERIA:
            self.add_criteria_row()
            row_idx = len(self.criteria_rows) - 1
            row_data = self.criteria_rows[row_idx]
            row_data["name_entry"].insert(0, criteria["name"])
            row_data["weight_entry"].insert(0, str(criteria["weight"]))
            if criteria["direction"] == "Minimizēt":
                row_data["direction_var"].set("Minimizēt")
        
        # Add/Remove buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=11, column=0, columnspan=3, pady=10, sticky='w')
        
        ttk.Button(button_frame, text="Pievienot kritēriju", command=self.add_criteria_row).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Dzēst pēdējo", command=self.remove_criteria_row).pack(side=tk.LEFT)
        
        # Navigation buttons
        nav_frame = ttk.Frame(frame)
        nav_frame.grid(row=12, column=0, columnspan=3, pady=20, sticky='e')
        
        ttk.Button(nav_frame, text="Tālāk >", command=lambda: self.notebook.select(1)).pack(side=tk.RIGHT)
    
    def add_criteria_row(self):
        """Add a new row for criteria input"""
        row = len(self.criteria_rows)
        
        # Create widgets for this row
        name_entry = ttk.Entry(self.scrollable_criteria_frame, width=40)
        weight_entry = ttk.Entry(self.scrollable_criteria_frame, width=10)
        
        # Direction selection (maximize or minimize)
        direction_var = tk.StringVar(value="max")
        direction_combo = ttk.Combobox(self.scrollable_criteria_frame, 
                                       textvariable=direction_var,
                                       values=["Maksimizēt", "Minimizēt"],
                                       state="readonly",
                                       width=15)
        direction_combo.current(0)
        
        # Place widgets in grid
        name_entry.grid(row=row, column=0, padx=(0, 10), pady=2, sticky='w')
        weight_entry.grid(row=row, column=1, padx=(0, 10), pady=2)
        direction_combo.grid(row=row, column=2, pady=2, sticky='w')
        
        # Save references
        self.criteria_rows.append({
            "name_entry": name_entry,
            "weight_entry": weight_entry,
            "direction_var": direction_var
        })
    
    def remove_criteria_row(self):
        """Remove the last criteria row"""
        if len(self.criteria_rows) > 1:  # Keep at least one row
            # Get the last row
            row_data = self.criteria_rows.pop()
            
            # Destroy widgets
            row_data["name_entry"].destroy()
            row_data["weight_entry"].destroy()
            
            # Get the combobox widget and destroy it
            for widget in self.scrollable_criteria_frame.grid_slaves():
                if isinstance(widget, ttk.Combobox) and widget.grid_info()["row"] == len(self.criteria_rows):
                    widget.destroy()
                    break
    
    def setup_options_tab(self):
        """Setup the integration options tab"""
        frame = ttk.Frame(self.options_tab, padding="10 10 10 10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Integrācijas opciju ģenerēšana", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        info_text = ("Definējiet iespējamās integrācijas stratēģijas (opcijas). "
                    "Katrai opcijai norādiet nosaukumu un aprakstu.")
        ttk.Label(frame, text=info_text, wraplength=800).grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky='w')
        
        # Common integration strategies
        ttk.Label(frame, text="Tipiskās integrācijas stratēģijas:", font=("Arial", 10, "bold")).grid(row=2, column=0, columnspan=2, pady=(10, 5), sticky='w')
        strategies = [
            "• Līdzāspastāvēšana - minimāla integrācija, sistēmas darbojas atsevišķi, dati apmainīti periodiski", 
            "• Daļēja integrācija - kritisko datu sinhronizācija starp sistēmām, pārējais atstāts atsevišķi", 
            "• Pilnīga integrācija - visu sistēmu apvienošana vienotā platformā",
            "• Pakāpeniska integrācija - sākotnēji daļēja integrācija ar plānu pāriet uz pilnīgu"
        ]
        
        for i, strategy in enumerate(strategies):
            ttk.Label(frame, text=strategy, wraplength=800).grid(row=3+i, column=0, columnspan=2, sticky='w')
        
        # Options input section
        ttk.Label(frame, text="Opcijas nosaukums", font=("Arial", 10, "bold")).grid(row=8, column=0, pady=(20, 5), sticky='w')
        ttk.Label(frame, text="Apraksts", font=("Arial", 10, "bold")).grid(row=8, column=1, pady=(20, 5), sticky='w')
        
        # Frame for option entries (scrollable)
        self.option_entries_frame = ttk.Frame(frame)
        self.option_entries_frame.grid(row=9, column=0, columnspan=2, sticky='nsew')
        
        # Scrollable canvas for option entries
        canvas = tk.Canvas(self.option_entries_frame)
        scrollbar = ttk.Scrollbar(self.option_entries_frame, orient="vertical", command=canvas.yview)
        self.scrollable_options_frame = ttk.Frame(canvas)
        
        self.scrollable_options_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_options_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initial option rows
        self.option_rows = []
        for i in range(3):  # Start with 3 empty option rows
            self.add_option_row()
        
        # Add example options
        example_options = [
            ("Līdzāspastāvēšana", "Minimāla integrācija, sistēmas darbojas atsevišķi, dati apmainīti periodiski"),
            ("Daļēja integrācija", "Kritisko datu sinhronizācija starp sistēmām, pārējais atstāts atsevišķi"),
            ("Pilnīga integrācija", "Visu sistēmu apvienošana vienotā platformā")
        ]
        
        for i, (name, desc) in enumerate(example_options):
            if i < len(self.option_rows):
                self.option_rows[i]["name_entry"].insert(0, name)
                self.option_rows[i]["desc_text"].insert("1.0", desc)
        
        # Add/Remove buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=10, sticky='w')
        
        ttk.Button(button_frame, text="Pievienot opciju", command=self.add_option_row).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Dzēst pēdējo", command=self.remove_option_row).pack(side=tk.LEFT)
        
        # Navigation buttons
        nav_frame = ttk.Frame(frame)
        nav_frame.grid(row=11, column=0, columnspan=2, pady=20, sticky='e')
        
        ttk.Button(nav_frame, text="< Atpakaļ", command=lambda: self.notebook.select(0)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(nav_frame, text="Tālāk >", command=lambda: self.notebook.select(2)).pack(side=tk.LEFT)
    
    def add_option_row(self):
        """Add a new row for option input"""
        row = len(self.option_rows)
        
        # Create widgets for this row
        name_entry = ttk.Entry(self.scrollable_options_frame, width=30)
        desc_text = scrolledtext.ScrolledText(self.scrollable_options_frame, width=50, height=3, wrap=tk.WORD)
        
        # Place widgets in grid
        name_entry.grid(row=row, column=0, padx=(0, 10), pady=5, sticky='nw')
        desc_text.grid(row=row, column=1, pady=5, sticky='w')
        
        # Save references
        self.option_rows.append({
            "name_entry": name_entry,
            "desc_text": desc_text
        })
    
    def remove_option_row(self):
        """Remove the last option row"""
        if len(self.option_rows) > 1:  # Keep at least one row
            # Get the last row
            row_data = self.option_rows.pop()
            
            # Destroy widgets
            row_data["name_entry"].destroy()
            row_data["desc_text"].destroy()
    
    def setup_evaluation_tab(self):
        """Setup the evaluation tab"""
        frame = ttk.Frame(self.evaluation_tab, padding="10 10 10 10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Novērtēšana", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        info_text = ("Novērtējiet katru opciju pēc katra kritērija. "
                    "Ievadiet vērtības atbilstoši kritērija tipam - jo lielāka vērtība, jo labāka opcija "
                    "(maksimizējamiem kritērijiem), vai jo mazāka vērtība, jo labāka opcija (minimizējamiem kritērijiem).")
        ttk.Label(frame, text=info_text, wraplength=800).grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky='w')
        
        # Evaluation matrix is created dynamically when this tab is selected
        self.evaluation_matrix_frame = ttk.Frame(frame)
        self.evaluation_matrix_frame.grid(row=2, column=0, columnspan=2, sticky='nsew')
        
        # Navigation buttons
        nav_frame = ttk.Frame(frame)
        nav_frame.grid(row=3, column=0, columnspan=2, pady=20, sticky='e')
        
        ttk.Button(nav_frame, text="< Atpakaļ", command=lambda: self.notebook.select(1)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(nav_frame, text="Aprēķināt rezultātus", command=self.calculate_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(nav_frame, text="Tālāk >", command=lambda: self.notebook.select(3)).pack(side=tk.LEFT)
    
    def setup_results_tab(self):
        """Setup the results tab"""
        frame = ttk.Frame(self.results_tab, padding="10 10 10 10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Lēmumu atbalsts", font=("Arial", 14, "bold")).grid(row=0, column=0, pady=10, sticky='w')
        
        # Create a notebook for results sections
        self.results_notebook = ttk.Notebook(frame)
        self.results_notebook.grid(row=1, column=0, pady=(0, 20), sticky='nsew')
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_tab, text="Kopsavilkums")
        
        # Calculations tab
        self.calculations_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.calculations_tab, text="Aprēķinu detaļas")
        
        # Results text area in summary tab
        self.results_text = scrolledtext.ScrolledText(self.summary_tab, width=80, height=15, wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True, pady=(10, 10))
        
        # Calculations text area in calculations tab
        self.calc_text = scrolledtext.ScrolledText(self.calculations_tab, width=80, height=20, wrap=tk.WORD)
        self.calc_text.pack(fill='both', expand=True, pady=(10, 10))
        
        # Frame for chart
        self.chart_frame = ttk.Frame(frame)
        self.chart_frame.grid(row=2, column=0, pady=(0, 20), sticky='nsew')
        
        # Navigation buttons
        nav_frame = ttk.Frame(frame)
        nav_frame.grid(row=3, column=0, pady=10, sticky='e')
        
        ttk.Button(nav_frame, text="< Atpakaļ", command=lambda: self.notebook.select(2)).pack(side=tk.LEFT)
    
    def on_tab_changed(self, event):
        """Handle tab change events"""
        selected_tab = self.notebook.index(self.notebook.select())
        
        if selected_tab == 2:  # Evaluation tab
            self.load_evaluation_matrix()
    
    def validate_criteria(self):
        """Validate criteria input and normalize weights"""
        self.criteria = []
        self.criteria_weights = []
        criteria_directions = []
        
        # Extract data from UI
        for row_data in self.criteria_rows:
            name = row_data["name_entry"].get().strip()
            weight_str = row_data["weight_entry"].get().strip()
            direction = row_data["direction_var"].get()
            
            if name and weight_str:
                try:
                    weight = float(weight_str)
                    if weight <= 0:
                        messagebox.showwarning("Validācijas kļūda", f"Kritērijam '{name}' svaram jābūt pozitīvam skaitlim.")
                        return False
                    
                    self.criteria.append(name)
                    self.criteria_weights.append(weight)
                    criteria_directions.append(1 if direction == "Maksimizēt" else -1)
                    
                except ValueError:
                    messagebox.showwarning("Validācijas kļūda", f"Kritērijam '{name}' svaram jābūt skaitlim.")
                    return False
        
        if not self.criteria:
            messagebox.showwarning("Validācijas kļūda", "Lūdzu, norādiet vismaz vienu kritēriju.")
            return False
        
        # Normalize weights to sum to 1
        total_weight = sum(self.criteria_weights)
        self.criteria_weights = [w/total_weight for w in self.criteria_weights]
        
        # Save directions (1 for maximize, -1 for minimize)
        self.criteria_directions = criteria_directions
        
        return True
    
    def validate_options(self):
        """Validate options input"""
        self.options = []
        self.option_descriptions = []
        
        # Extract data from UI
        for row_data in self.option_rows:
            name = row_data["name_entry"].get().strip()
            description = row_data["desc_text"].get("1.0", tk.END).strip()
            
            if name:
                self.options.append(name)
                self.option_descriptions.append(description)
        
        if not self.options:
            messagebox.showwarning("Validācijas kļūda", "Lūdzu, norādiet vismaz vienu integrācijas opciju.")
            return False
        
        return True
    
    def load_evaluation_matrix(self):
        """Create and populate the evaluation matrix"""
        # First validate the criteria and options
        if not self.validate_criteria() or not self.validate_options():
            self.notebook.select(0 if not self.validate_criteria() else 1)
            return
        
        # Clear previous matrix
        for widget in self.evaluation_matrix_frame.winfo_children():
            widget.destroy()
        
        # Create header row
        ttk.Label(self.evaluation_matrix_frame, text="Opcija / Kritērijs", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        for col, criterion in enumerate(self.criteria):
            direction = "↑" if self.criteria_directions[col] == 1 else "↓"
            header_text = f"{criterion} ({direction})"
            ttk.Label(self.evaluation_matrix_frame, text=header_text, font=("Arial", 10, "bold")).grid(row=0, column=col+1, padx=5, pady=5)
        
        # Initialize evaluation matrix if it's empty
        if not self.evaluations or len(self.evaluations) != len(self.options) or len(self.evaluations[0]) != len(self.criteria):
            self.evaluations = [[0 for _ in range(len(self.criteria))] for _ in range(len(self.options))]
        
        # Create matrix entry grid
        self.evaluation_entries = []
        for row, option in enumerate(self.options):
            # Option name in first column
            ttk.Label(self.evaluation_matrix_frame, text=option).grid(row=row+1, column=0, padx=5, pady=2, sticky='w')
            
            # Create entries for each criterion
            row_entries = []
            for col in range(len(self.criteria)):
                entry = ttk.Entry(self.evaluation_matrix_frame, width=10)
                entry.grid(row=row+1, column=col+1, padx=5, pady=2)
                
                # Pre-fill with existing evaluation data if available
                if self.evaluations[row][col]:
                    entry.insert(0, str(self.evaluations[row][col]))
                
                row_entries.append(entry)
            
            self.evaluation_entries.append(row_entries)
    
    def calculate_results(self):
        """Calculate results based on evaluations"""
        # Get evaluations from UI
        try:
            evaluations = []
            for row in range(len(self.options)):
                row_values = []
                for col in range(len(self.criteria)):
                    value_str = self.evaluation_entries[row][col].get().strip()
                    if not value_str:
                        messagebox.showwarning("Validācijas kļūda", f"Lūdzu, aizpildiet visas vērtības novērtēšanas matricā.")
                        return False
                    
                    try:
                        value = float(value_str)
                        row_values.append(value)
                    except ValueError:
                        messagebox.showwarning("Validācijas kļūda", 
                                              f"Opcijas '{self.options[row]}' vērtībai pēc kritērija '{self.criteria[col]}' jābūt skaitlim.")
                        return False
                
                evaluations.append(row_values)
            
            self.evaluations = evaluations
        except:
            messagebox.showwarning("Kļūda", "Notika kļūda, iegūstot novērtējumus. Lūdzu, pārbaudiet ievadītās vērtības.")
            return False
        
        # Normalize evaluations
        self.normalize_evaluations()
        
        # Calculate weighted scores
        self.calculate_weighted_scores()
        
        # Calculate final scores
        self.calculate_final_scores()
        
        # Display results
        self.display_results()
        
        # Go to results tab
        self.notebook.select(3)
        
        return True
    
    def normalize_evaluations(self):
        """Normalize the evaluation matrix"""
        # Convert to numpy array for easier calculations
        eval_array = np.array(self.evaluations)
        norm_array = np.zeros_like(eval_array, dtype=float)
        
        # Normalize each criterion column
        for j in range(len(self.criteria)):
            direction = self.criteria_directions[j]
            column = eval_array[:, j].astype(float)
            
            # For maximize criteria (direction=1), higher values are better
            # For minimize criteria (direction=-1), lower values are better
            if direction == 1:  # Maximize
                min_val = np.min(column)
                max_val = np.max(column)
                
                if max_val == min_val:
                    norm_array[:, j] = 1.0  # All options are equally good
                else:
                    # Normalize to [0, 1] where 1 is best
                    norm_array[:, j] = (column - min_val) / (max_val - min_val)
            else:  # Minimize
                min_val = np.min(column)
                max_val = np.max(column)
                
                if max_val == min_val:
                    norm_array[:, j] = 1.0  # All options are equally good
                else:
                    # Normalize to [0, 1] where 1 is best (invert min/max)
                    norm_array[:, j] = (max_val - column) / (max_val - min_val)
        
        self.normalized_evaluations = norm_array.tolist()
    
    def calculate_weighted_scores(self):
        """Calculate weighted scores"""
        # Convert to numpy arrays
        norm_array = np.array(self.normalized_evaluations)
        weights = np.array(self.criteria_weights)
        
        # Multiply each normalized value by the corresponding weight
        weighted = norm_array * weights
        
        self.weighted_scores = weighted.tolist()
    
    def calculate_final_scores(self):
        """Calculate final scores for each option"""
        # Sum weighted scores across all criteria for each option
        weighted_array = np.array(self.weighted_scores)
        final_scores = np.sum(weighted_array, axis=1)
        
        self.final_scores = final_scores.tolist()
    
    def display_results(self):
        """Display results in the results tab"""
        # Clear previous results
        self.results_text.delete("1.0", tk.END)
        self.calc_text.delete("1.0", tk.END)
        
        # Find the best option
        best_option_index = np.argmax(self.final_scores)
        best_option = self.options[best_option_index]
        best_score = self.final_scores[best_option_index]
        
        # Format final scores as percentages
        score_percentages = [score * 100 for score in self.final_scores]
        
        # Prepare results output
        results = f"REKOMENDĀCIJA\n"
        results += f"=============\n\n"
        results += f"Ieteiktā opcija: {best_option}\n"
        results += f"Opcijas kopējais novērtējums: {best_score:.2f} ({score_percentages[best_option_index]:.1f}%)\n\n"
        
        # Add ranking of options
        results += "OPCIJU RANGS\n"
        results += "============\n\n"
        
        # Sort options by score
        ranked_indices = np.argsort(self.final_scores)[::-1]  # Descending order
        
        for i, idx in enumerate(ranked_indices):
            results += f"{i+1}. {self.options[idx]}: {self.final_scores[idx]:.2f} ({score_percentages[idx]:.1f}%)\n"
        
        # Add strengths/weaknesses of the best option
        results += f"\nPAMATOJUMS\n"
        results += f"==========\n\n"
        
        # Find the strongest and weakest criteria for the best option
        normalized_scores = np.array(self.normalized_evaluations[best_option_index])
        strongest_idx = np.argmax(normalized_scores)
        weakest_idx = np.argmin(normalized_scores)
        
        strengths = []
        weaknesses = []
        
        # Add significant strengths (above 0.7 normalized score)
        for i, score in enumerate(normalized_scores):
            if score > 0.7:
                strengths.append(self.criteria[i])
            elif score < 0.3:
                weaknesses.append(self.criteria[i])
        
        if strengths:
            results += f"Stiprās puses: {', '.join(strengths)}\n"
        
        if weaknesses:
            results += f"Vājās puses: {', '.join(weaknesses)}\n"
        
        # Add best option explanation
        results += f"\nOpcija '{best_option}' ir rekomendējama, jo tā kopumā "
        results += f"vislabāk atbilst definētajiem kritērijiem. "
        
        if strengths:
            results += f"Tā īpaši izceļas ar: {', '.join(strengths)}. "
        
        # Compare to second best if there is one
        if len(ranked_indices) > 1:
            second_best_idx = ranked_indices[1]
            second_best = self.options[second_best_idx]
            diff = score_percentages[best_option_index] - score_percentages[second_best_idx]
            
            if diff < 5:
                results += f"\n\nTomēr '{second_best}' arī ir laba alternatīva, jo tās kopējais novērtējums "
                results += f"atpaliek tikai par {diff:.1f}% punktiem."
            else:
                results += f"\n\nSalīdzinājumā ar '{second_best}' opciju, rekomendētā opcija ir par {diff:.1f}% punktiem labāka."
        
        # Add description of the best option if available
        if self.option_descriptions[best_option_index]:
            results += f"\n\nRekomendētās opcijas apraksts:\n{self.option_descriptions[best_option_index]}"
        
        # Display results
        self.results_text.insert(tk.END, results)
        
        # Create and display calculation details
        self.display_calculation_details()
        
        # Create and display chart
        self.create_results_chart()
    
    def display_calculation_details(self):
        """Display detailed calculation steps in the calculation tab"""
        calc_details = f"APRĒĶINU DETAĻAS\n"
        calc_details += f"================\n\n"
        
        # 1. Show criteria and weights
        calc_details += f"1. KRITĒRIJI UN SVARI\n"
        calc_details += f"---------------------\n\n"
        
        for i, criterion in enumerate(self.criteria):
            direction = "Maksimizēt" if self.criteria_directions[i] == 1 else "Minimizēt"
            calc_details += f"{criterion}: svars = {self.criteria_weights[i]:.2f} ({self.criteria_weights[i]*100:.1f}%), mērķis = {direction}\n"
        
        # 2. Show original evaluation matrix
        calc_details += f"\n\n2. SĀKOTNĒJĀ NOVĒRTĒŠANAS MATRICA\n"
        calc_details += f"----------------------------------\n\n"
        
        # Add header row
        calc_details += f"{'Opcija':<20} | "
        for criterion in self.criteria:
            calc_details += f"{criterion:<15} | "
        calc_details += "\n"
        calc_details += "-" * (20 + (15 + 3) * len(self.criteria)) + "\n"
        
        # Add data rows
        for i, option in enumerate(self.options):
            calc_details += f"{option:<20} | "
            for j, criterion in enumerate(self.criteria):
                calc_details += f"{self.evaluations[i][j]:<15.2f} | "
            calc_details += "\n"
        
        # 3. Explain normalization process
        calc_details += f"\n\n3. NORMALIZĀCIJA\n"
        calc_details += f"------------------\n\n"
        
        calc_details += f"Normalizācija pārveido visu kritēriju vērtības uz skalu [0, 1], kur:\n"
        calc_details += f"- Maksimizējamiem kritērijiem: 1 = labākā vērtība, 0 = sliktākā vērtība\n"
        calc_details += f"- Minimizējamiem kritērijiem: 1 = labākā vērtība (zemākā sākotnējā), 0 = sliktākā vērtība (augstākā sākotnējā)\n\n"
        
        # Show normalization formulas
        calc_details += f"Normalizācijas formulas:\n"
        calc_details += f"- Maksimizējamiem kritērijiem: (vērtība - min) / (max - min)\n"
        calc_details += f"- Minimizējamiem kritērijiem: (max - vērtība) / (max - min)\n\n"
        
        # 4. Show normalized matrix
        calc_details += f"\n4. NORMALIZĒTĀ MATRICA\n"
        calc_details += f"----------------------\n\n"
        
        # Add header row
        calc_details += f"{'Opcija':<20} | "
        for criterion in self.criteria:
            calc_details += f"{criterion:<15} | "
        calc_details += "\n"
        calc_details += "-" * (20 + (15 + 3) * len(self.criteria)) + "\n"
        
        # Add data rows
        for i, option in enumerate(self.options):
            calc_details += f"{option:<20} | "
            for j, criterion in enumerate(self.criteria):
                calc_details += f"{self.normalized_evaluations[i][j]:<15.4f} | "
            calc_details += "\n"
        
        # 5. Explain weighted scores
        calc_details += f"\n\n5. SVĒRTĀS VĒRTĪBAS\n"
        calc_details += f"--------------------\n\n"
        
        calc_details += f"Svērtās vērtības tiek aprēķinātas, reizinot normalizētās vērtības ar kritēriju svariem.\n"
        calc_details += f"Formula: svērtā_vērtība = normalizētā_vērtība × kritērija_svars\n\n"
        
        # 6. Show weighted matrix
        calc_details += f"\n6. SVĒRTĀ MATRICA\n"
        calc_details += f"-----------------\n\n"
        
        # Add header row
        calc_details += f"{'Opcija':<20} | "
        for criterion in self.criteria:
            calc_details += f"{criterion:<15} | "
        calc_details += f"{'KOPĀ':<15} | "
        calc_details += "\n"
        calc_details += "-" * (20 + (15 + 3) * (len(self.criteria) + 1)) + "\n"
        
        # Add data rows
        for i, option in enumerate(self.options):
            calc_details += f"{option:<20} | "
            for j, criterion in enumerate(self.criteria):
                calc_details += f"{self.weighted_scores[i][j]:<15.4f} | "
            calc_details += f"{self.final_scores[i]:<15.4f} | "
            calc_details += "\n"
        
        # 7. Final scores explanation
        calc_details += f"\n\n7. GALA REZULTĀTS\n"
        calc_details += f"------------------\n\n"
        
        calc_details += f"Gala rezultāts katrai opcijai tiek aprēķināts, summējot visas svērtās vērtības pa kritērijiem.\n"
        calc_details += f"Formula: gala_rezultāts = ∑(svērtās_vērtības)\n\n"
        
        for i, option in enumerate(self.options):
            calc_details += f"{option}: {self.final_scores[i]:.4f} ({self.final_scores[i]*100:.1f}%)\n"
        
        # Display calculation details
        self.calc_text.insert(tk.END, calc_details)
    
    def create_results_chart(self):
        """Create and display chart of final scores"""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Create figure with more height to accommodate labels
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Sort options by score
        sorted_indices = np.argsort(self.final_scores)[::-1]
        sorted_options = [self.options[i] for i in sorted_indices]
        sorted_scores = [self.final_scores[i] * 100 for i in sorted_indices]  # Convert to percentages
        
        # Create bar chart
        bars = ax.bar(range(len(sorted_options)), sorted_scores, color='skyblue')
        
        # Highlight the best option
        best_idx = np.argmax(sorted_scores)
        bars[best_idx].set_color('green')
        
        # Add labels and title
        ax.set_ylabel('Kopējais novērtējums (%)')
        ax.set_title('Integrācijas opciju salīdzinājums')
        
        # Add score values on top of bars
        for i, v in enumerate(sorted_scores):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        # Format chart
        ax.set_xticks(range(len(sorted_options)))
        ax.set_xticklabels(sorted_options, rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(bottom=0.2)
        fig.tight_layout()
        
        # Embed chart in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = IntegrationDecisionTool(root)
    root.mainloop() 