# import os
# import graphviz

# # PATH Injection (Apne PC ka path verify kar lena)
# # If this fails, make sure Graphviz is installed and 'dot' is in your system PATH
# os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# if not os.path.exists('analysis'): os.makedirs('analysis')

# # Creating the Digraph - Setting DPI and Base Size for high-resolution output
# # dpi='300' ensures print-quality resolution. size adjusts the canvas.
# dot = graphviz.Digraph('ALS_Master_Pipeline_v3', comment='Professional Workflow')

# # Global Stylings - Setting High DPI and optimal size
# dot.attr(rankdir='TB', size='12,16', ratio='compress', dpi='300', fontname='Segoe UI, Arial, Helvetica, sans-serif')
# dot.attr('node', shape='box', style='filled,rounded', color='#2c3e50', 
#          fontname='Segoe UI, Arial, Helvetica, sans-serif', fontsize='12', margin='0.3', penwidth='1')
# dot.attr('edge', color='#34495e', penwidth='1.5', arrowhead='vee')

# # --- PHASE I: DATA HARVESTING (ORANGE) ---
# with dot.subgraph(name='cluster_0') as c:
#     c.attr(label='PHASE I: DATA ACQUISITION & REFINEMENT', style='filled', color='#FFF3E0', fontcolor='#E65100', fontsize='15', fontname='Segoe UI, Arial, Helvetica, sans-serif bold')
    
#     c.node('A', 'PRO-ACT Database\n(Initial Registry: ~10,700 Patients)', shape='database', fillcolor='#FFE0B2')
#     c.node('B', 'Filtering & Integration:\n1. 7 Clinical Tables Selected (ALSFRS, FVC, etc.)\n2. 30% Records Dropped (Insufficient Longitudinal Data)', fillcolor='#FFE0B2')
#     c.node('C', 'Final Study Cohort\n(~3,687 Patients | Cleaned Data)', shape='parallelogram', fillcolor='#FFCC80')
#     c.node('D', 'Pre-processing Pipeline:\n• IQR Outlier Removal\n• Median Imputation\n• Min-Max Normalization', style='dashed,filled', fillcolor='#FFF3E0')
    
#     c.edge('A', 'B')
#     c.edge('B', 'C')
#     c.edge('C', 'D')

# # --- PHASE II: FEATURE TOURNAMENT (BLUE) ---
# with dot.subgraph(name='cluster_1') as c:
#     c.attr(label='PHASE II: FEATURE SELECTION TOURNAMENT', style='filled', color='#E3F2FD', fontcolor='#0D47A1', fontsize='15', fontname='Segoe UI, Arial, Helvetica, sans-serif bold')
    
#     c.node('E', 'Tournament Environment:\nRF vs XGBoost vs Linear Regression', shape='ellipse', fillcolor='#BBDEFB')
#     c.node('F', 'Evaluation Metric:\nPCC Score vs Model Stability', fillcolor='#BBDEFB')
#     c.node('G', 'Feature Selection Winner:\nElite-5 Features (via RF Importance)\n[Onset, ALSFRS, FVC, Weight, Speech]', fillcolor='#64B5F6', fontcolor='white')
    
#     c.edge('E', 'F')
#     c.edge('F', 'G')

# # --- PHASE III: ANFIS ARCHITECTURE (GREEN) 
# with dot.subgraph(name='cluster_2') as c:
#     c.attr(label='PHASE III: PROPOSED ANFIS MODELING', style='filled', color='#E8F5E9', fontcolor='#1B5E20', fontsize='15', fontname='Segoe UI, Arial, Helvetica, sans-serif bold')
    
#     c.node('H', 'Neuro-Fuzzy Setup:\n70-30 Training/Testing Split', shape='diamond', fillcolor='#C8E6C9')
#     c.node('I', 'Fuzzification & Rule Generation:\nGrid Partitioning (243 Rules)\nGaussian Membership Functions', fillcolor='#C8E6C9')
#     c.node('J', 'Hybrid Learning Optimization:\nBackpropagation + Least Squares', fillcolor='#81C784', fontcolor='black')
#     c.node('K', 'Prediction Engine:\nDefuzzified Slope Estimation', shape='parallelogram', fillcolor='#C8E6C9')
    
#     c.edge('H', 'I')
#     c.edge('I', 'J')
#     c.edge('J', 'K')

# # --- PHASE IV: EVALUATION & XAI (PURPLE) ---
# with dot.subgraph(name='cluster_3') as c:
#     c.attr(label='PHASE IV: VALIDATION & EXPLAINABILITY', style='filled', color='#F3E5F5', fontcolor='#4A148C', fontsize='15', fontname='Segoe UI, Arial, Helvetica, sans-serif bold')
    
#     c.node('L', 'Performance Metrics:\nPCC Score: ~0.40 | RMSD Error: 0.5292', fillcolor='#E1BEE7')
#     c.node('M', 'Explainable AI (XAI):\nData-Driven Rule Extraction\n(Top-10 Max Support Rules)', fillcolor='#CE93D8')
#     c.node('N', 'Comparative Analysis:\nProposed ANFIS vs DL Literature', shape='doubleoctagon', fillcolor='#AB47BC', fontcolor='white')
    
#     c.edge('L', 'M')
#     c.edge('M', 'N')

# # --- CONNECTING THE CLUSTERS ---
# dot.edge('D', 'E', label='Cleaned Data')
# dot.edge('G', 'H', label='Elite Features Selected')
# dot.edge('K', 'L', label='Prediction Output')

# # Save and Render
# output_file = 'analysis/final_als_pipeline_v3_highres'
# # Render as PNG, cleanup temporary dot file
# dot.render(output_file, format='png', cleanup=True)

# print(f" flowchart check ready: {output_file}.png")