import os
import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
from matplotlib.colors import to_rgb, to_hex
import colorsys

ecod_redundancy = 'F100'
# ecod_redundancy = 'F40'

# === File Paths ===
data_path = f'../data/{ecod_redundancy}'
output_path = '../outputs'

dataset_path = os.path.join(output_path, 'tsne_results.pkl')
folds_dataset_path = os.path.join(data_path, 'xfolds.csv')
domain_dataset_path = os.path.join(data_path, 'domains.csv')
architecture_color_dataset_path = os.path.join(data_path, 'architecture_colors.tsv')

# === Load Data ===
with open(dataset_path, 'rb') as pickle_file:
    dataset = pickle.load(pickle_file)

folds_dataset = pd.read_csv(folds_dataset_path)
fold_architecture_map = dict(zip(folds_dataset['fold'], folds_dataset['architecture']))

domain_dataset = pd.read_csv(domain_dataset_path, dtype={'domain_id': str}, usecols=['domain_id', 'domain_name'])
domain_name_map = dict(zip(domain_dataset['domain_id'], domain_dataset['domain_name']))

architecture_color_dataset = pd.read_csv(architecture_color_dataset_path, sep='\t', names=['architecture', 'color'])
architecture_color_map = dict(zip(architecture_color_dataset['architecture'], architecture_color_dataset['color']))

xfolds = dataset['xfold']
tsne_results = dataset['tsne_results']
domain_ids = dataset['domain_id']

modalities = ['sequence'] * len(xfolds) + ['stretch'] * len(xfolds) + ['structure'] * len(xfolds)
xfolds = np.tile(xfolds, 3)
domain_ids_combined = np.tile(domain_ids, 3)

df = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'modality': modalities,
    'fold': xfolds,
    'domain_id': domain_ids_combined,
})

df['architecture'] = df['fold'].map(fold_architecture_map)
df['domain_name'] = df['domain_id'].map(domain_name_map)

marker_symbols = {
    'sequence': 'circle',
    'stretch': 'x',
    'structure': 'cross'
}

def create_plot(modality_filter_list, search_query, color_level, architecture_filter, fold_filter):
    def generate_fold_color_map():
        fold_color_map = {}
        filtered_folds = set(df['fold']) if not fold_filter else set(fold_filter)
        if fold_filter:
            # Assign random distinct colors when specific folds are selected
            import matplotlib.pyplot as plt
            colors = plt.cm.tab20(np.linspace(0, 1, len(filtered_folds)))
            for i, fold in enumerate(sorted(filtered_folds)):
                fold_color_map[fold] = to_hex(colors[i % len(colors)])
        else:
            for fold, arch in fold_architecture_map.items():
                base_color = architecture_color_map.get(arch, '#888888')
                r, g, b = to_rgb(base_color)
                h, l, s = colorsys.rgb_to_hls(r, g, b)
                l = min(1.0, max(0.2, l + np.random.uniform(-0.2, 0.2)))
                r_shift, g_shift, b_shift = colorsys.hls_to_rgb(h, l, s)
                fold_color_map[fold] = to_hex((r_shift, g_shift, b_shift))
        return fold_color_map

    fold_color_map = generate_fold_color_map() if color_level == 'fold' else None
    if not modality_filter_list:
        return go.Figure(), ""

    filtered_df = df[df['modality'].isin(modality_filter_list)].copy()
    if architecture_filter:
        filtered_df = filtered_df[filtered_df['architecture'].isin(architecture_filter)]
    if fold_filter:
        filtered_df = filtered_df[filtered_df['fold'].isin(fold_filter)]
    filtered_df['highlight'] = False
    filtered_df['opacity'] = 0.8
    filtered_df['size'] = 6
    match_count = 0

    if search_query:
        search_query = search_query.lower()

        def is_match(row):
            return (
                search_query in str(row['domain_name']).lower() or
                search_query in str(row['domain_id']).lower() or
                search_query in str(row['fold']).lower() or
                search_query in str(row['architecture']).lower()
            )

        filtered_df['highlight'] = filtered_df.apply(is_match, axis=1)
        match_count = filtered_df['highlight'].sum()
        filtered_df['opacity'] = np.where(filtered_df['highlight'], 1.0, 0.1)
        filtered_df['size'] = np.where(filtered_df['highlight'], 16, 4)

    fig = go.Figure()
    for (arch, mod), group in filtered_df.groupby([color_level, 'modality']):
        fig.add_trace(go.Scattergl(
            x=group['x'],
            y=group['y'],
            mode='markers',
            name=f"{group[color_level].iloc[0]}, {mod}",
            marker=dict(
                size=group['size'],
                opacity=group['opacity'],
                symbol=marker_symbols.get(mod, 'circle'),
                color=architecture_color_map.get(group[color_level].iloc[0], '#888888') if color_level == 'architecture' else fold_color_map.get(group[color_level].iloc[0], '#888888'),
                line=dict(width=0)
            ),
            hovertext=group['domain_name'],
            customdata=group[['domain_name', 'domain_id', 'architecture', 'fold']],
            hovertemplate="<br>".join([
                "<b>%{customdata[0]}</b>",
                "ID: %{customdata[1]}",
                "Architecture: %{customdata[2]}",
                "Fold: %{customdata[3]}<extra></extra>"
            ])
        ))

    fig.update_layout(
        title="t-SNE Visualization of ECOD Domains by Modality",
        uirevision='static',
        hovermode='closest',
        autosize=True,
        margin=dict(l=10, r=10, t=50, b=10),
        dragmode='pan',
    )

    match_text = f"Highlighted {match_count} result(s)." if search_query else ""
    return fig, match_text

# === Initialize Dash App ===
app = dash.Dash(__name__)

# === Layout ===
app.layout = html.Div(
    style={
        'height': '100vh',
        'width': '100vw',
        'display': 'flex',
        'flexDirection': 'column',
        'padding': '0',
        'margin': '0',
        'position': 'absolute',
        'top': 0,
        'bottom': 0,
        'left': 0,
        'right': 0
    },
    children=[
        html.Div([
            html.H1("Interactive t-SNE Visualization", style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                dcc.Dropdown(
                    id='color-level-dropdown',
                    options=[
                        {'label': 'Architecture', 'value': 'architecture'},
                        {'label': 'Fold', 'value': 'fold'}
                    ],
                    value='architecture',
                    clearable=False,
                    style={'width': '220px'}
                ),
                dcc.Checklist(
                    id='modality-checklist',
                    options=[
                        {'label': 'Sequence', 'value': 'sequence'},
                        {'label': 'Stretch', 'value': 'stretch'},
                        {'label': 'Structure', 'value': 'structure'}
                    ],
                    value=['sequence', 'stretch', 'structure'],
                    labelStyle={'display': 'inline-block', 'marginRight': '10px'},
                    style={'margin': '0 10px'},
                    persistence=True,
                    persistence_type='session'
                ),
                dcc.Dropdown(
                    id='architecture-filter-dropdown',
                    options=[{'label': arch, 'value': arch} for arch in sorted(df['architecture'].dropna().unique())],
                    multi=True,
                    placeholder='Filter by architecture...',
                    style={'width': '220px'}
                ),
                dcc.Dropdown(
                    id='fold-filter-dropdown',
                    options=[{'label': f, 'value': f} for f in sorted(df['fold'].dropna().unique())],
                    multi=True,
                    placeholder='Filter by fold...',
                    style={'width': '220px'}
                ),
                dcc.Input(
                    id='search-input',
                    type='text',
                    placeholder='Search domain...',
                    debounce=True,
                    style={'width': '220px', 'padding': '8px', 'fontSize': '16px'}
                )
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '20px'}),
            html.Div(id='match-count-text', style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '16px', 'color': '#555'})
        ], style={'flex': '0 0 auto'}),

        html.Div([
            html.Div([
                html.H3("Selected Domains", style={'marginTop': '0', 'marginBottom': '10px'}),
                html.Button("Reset selection", id="reset-selection", style={'marginBottom': '10px'}),
                html.Pre(id='selected-info-content', style={'fontSize': '12px', 'overflowX': 'auto'})
            ], id='selected-info', style={
                'width': '220px',
                'borderRight': '1px solid #ccc',
                'padding': '10px',
                'overflowY': 'auto',
                'fontFamily': 'monospace'
            }),
            html.Div([
                dcc.Graph(id='tsne-plot', config={'scrollZoom': True,'displayModeBar': True,'doubleClick': 'reset'}, style={'height': '100%', 'width': '100%'})
            ], style={'flex': '1 1 auto'})
        ], style={'flex': '1 1 auto', 'display': 'flex', 'overflow': 'hidden'})
    ]
)

# === Callbacks ===
@app.callback(
    [Output('tsne-plot', 'figure'), Output('match-count-text', 'children'), Output('fold-filter-dropdown', 'options')],
    [Input('modality-checklist', 'value'), Input('search-input', 'value'), Input('color-level-dropdown', 'value'), Input('architecture-filter-dropdown', 'value'), Input('fold-filter-dropdown', 'value')]
)
def update_plot(selected_modalities, search_query, color_level, architecture_filter, fold_filter):
    if architecture_filter:
        fold_options = [{'label': f, 'value': f} for f, a in fold_architecture_map.items() if a in architecture_filter]
    else:
        fold_options = [{'label': f, 'value': f} for f in sorted(df['fold'].dropna().unique())]

    return create_plot(selected_modalities, search_query, color_level, architecture_filter, fold_filter) + (fold_options,)

@app.callback(
    Output('selected-info-content', 'children'),
    [Input('tsne-plot', 'clickData'), Input('reset-selection', 'n_clicks')],
    State('selected-info-content', 'children')
)
def update_selected_info(click_data, reset_clicks, current):
    triggered_id = ctx.triggered_id
    if triggered_id == 'reset-selection':
        return ""
    if click_data and 'customdata' in click_data['points'][0]:
        row = click_data['points'][0]['customdata']
        new_entry = {
            "domain_name": row[0],
            "domain_id": row[1],
            "architecture": row[2],
            "fold": row[3]
        }
        try:
            existing_data = json.loads(current) if current else []
        except Exception:
            existing_data = []
        if new_entry not in existing_data:
            existing_data.append(new_entry)
        return json.dumps(existing_data, indent=2)
    return current

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True)
