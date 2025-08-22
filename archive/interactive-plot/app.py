import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
from matplotlib.colors import to_rgb, to_hex
import colorsys
import matplotlib.pyplot as plt

# -------- You need to unzip data/F100/domains.zip before running this --------

ecod_redundancy = 'F100' # ----> We still need to populate the h_name, t_name, f_name columns in its domain.csv with real data
# ecod_redundancy = 'F40'  ----> This doesn't work correctly currently, we need to add the x_name, h_name, t_name, f_name columns to its domains.csv

# === File Paths ===
DATA_PATH = os.path.join('..', 'data', ecod_redundancy)
OUTPUT_PATH = os.path.join('..', 'outputs', ecod_redundancy)

DATASET_PATH = os.path.join(OUTPUT_PATH, 'tsne_results.pkl')
HIERARCHY_PATH = os.path.join(DATA_PATH, 'hierarchy.csv')
DOMAINS_PATH = os.path.join(DATA_PATH, 'domains.csv')
ARCH_COLOR_PATH = os.path.join(DATA_PATH, 'architecture_colors.tsv')

# === Load Data ===
with open(DATASET_PATH, 'rb') as pf:
    dataset = pickle.load(pf)

hierarchy_df = pd.read_csv(HIERARCHY_PATH)
hierarchy_df['fold'] = hierarchy_df['hierarchy'].str.split('.').str[0].astype(str)
fold_arch_map_original = dict(zip(hierarchy_df['fold'], hierarchy_df['architecture']))
hierarchy_to_xname_map = dict(zip(hierarchy_df['hierarchy'], hierarchy_df['fold_desc']))
hierarchy_to_hname_map = dict(zip(hierarchy_df['hierarchy'], hierarchy_df['h_desc']))
hierarchy_to_tname_map = dict(zip(hierarchy_df['hierarchy'], hierarchy_df['t_desc']))
hierarchy_to_fname_map = dict(zip(hierarchy_df['hierarchy'], hierarchy_df['f_desc']))

domain_df = pd.read_csv(DOMAINS_PATH, dtype={'domain_id': str}, usecols=['domain_id', 'domain_name', 'hierarchy'])
domain_name_map = dict(zip(domain_df['domain_id'], domain_df['domain_name']))
domain_df['x_name'] = domain_df['hierarchy'].map(hierarchy_to_xname_map)
domain_df['h_name'] = domain_df['hierarchy'].map(hierarchy_to_hname_map)
domain_df['t_name'] = domain_df['hierarchy'].map(hierarchy_to_tname_map)
domain_df['f_name'] = domain_df['hierarchy'].map(hierarchy_to_fname_map)

xname_map = dict(zip(domain_df['domain_id'], domain_df['x_name']))
hname_map = dict(zip(domain_df['domain_id'], domain_df['h_name']))
tname_map = dict(zip(domain_df['domain_id'], domain_df['t_name']))
fname_map = dict(zip(domain_df['domain_id'], domain_df['f_name']))

arch_color_df = pd.read_csv(ARCH_COLOR_PATH, sep='\t', names=['architecture', 'color'])
arch_color_map = dict(zip(arch_color_df['architecture'], arch_color_df['color']))

xfolds = dataset['xfold']
tsne = dataset['tsne_results']
domain_ids = dataset['domain_id']

# Create fold labels with x_name
fold_with_xname_map = {
    did: f"{fold} ({xname_map.get(did, '')})"
    for did, fold in zip(domain_ids, xfolds)
}
xfolds_with_xname = [fold_with_xname_map[did] for did in domain_ids]

# Updated fold_arch_map to match new fold labels
fold_arch_map = {
    fold_with_xname_map[did]: fold_arch_map_original.get(fold)
    for did, fold in zip(domain_ids, xfolds)
}

# Prepare repeated values
modalities = ['sequence'] * len(xfolds_with_xname) + ['stretch'] * len(xfolds_with_xname) + ['structure'] * len(xfolds_with_xname)
xfolds_rep = np.tile(xfolds_with_xname, 3)
domain_rep = np.tile(domain_ids, 3)

# Assemble DataFrame
df = pd.DataFrame({
    'x': tsne[:, 0], 'y': tsne[:, 1],
    'modality': modalities, 'fold': xfolds_rep,
    'domain_id': domain_rep
})

# print(fold_arch_map)
df['architecture'] = df['fold'].map(fold_arch_map)
df['domain_name'] = df['domain_id'].map(domain_name_map)
df['x_name'] = df['domain_id'].map(xname_map)
df['h_name'] = df['domain_id'].map(hname_map)
df['t_name'] = df['domain_id'].map(tname_map)
df['f_name'] = df['domain_id'].map(fname_map)

marker_symbols = {'sequence': 'circle', 'stretch': 'x', 'structure': 'cross'}

# (The imports and file loading sections are unchanged)

# === Plot creation function ===
def create_plot(modality_filter, search_query, color_level, arch_filter, fold_filter):
    def generate_fold_color_map():
        fold_color_map = {}
        active = set(fold_filter) if fold_filter else set(df['fold'])
        if fold_filter:
            colors = plt.cm.tab20(np.linspace(0, 1, len(active)))
            for i, f in enumerate(sorted(active)):
                fold_color_map[f] = to_hex(colors[i % len(colors)])
        else:
            for f, arch in fold_arch_map.items():
                base = arch_color_map.get(arch, '#888888')
                r, g, b = to_rgb(base)
                h, l, s = colorsys.rgb_to_hls(r, g, b)
                l = min(1.0, max(0.2, l + np.random.uniform(-0.2, 0.2)))
                fold_color_map[f] = to_hex(colorsys.hls_to_rgb(h, l, s))
        return fold_color_map

    fold_color_map = generate_fold_color_map() if color_level == 'fold' else None
    if not modality_filter:
        return go.Figure(), ""

    dff = df[df['modality'].isin(modality_filter)].copy()
    if arch_filter:
        dff = dff[dff['architecture'].isin(arch_filter)]
    if fold_filter:
        dff = dff[dff['fold'].isin(fold_filter)]

    dff['highlight'] = False
    dff['opacity'] = 0.8
    dff['size'] = 8
    match_count = 0

    if search_query:
        sq = search_query.lower()
        mask = dff.apply(lambda r: sq in str(r['domain_name']).lower() or
                                  sq in str(r['domain_id']).lower() or
                                  sq in str(r['fold']).lower() or
                                  sq in str(r['architecture']).lower() or
                                  sq in str(r['x_name']).lower() or
                                  sq in str(r['h_name']).lower() or
                                  sq in str(r['t_name']).lower() or
                                  sq in str(r['f_name']).lower(), axis=1)
        dff.loc[mask, ['highlight', 'opacity', 'size']] = [True, 1.0, 8]
        dff.loc[~mask, ['opacity', 'size']] = [0.6, 8]
        match_count = int(mask.sum())
    else:
        mask = dff.apply(lambda _: False, axis=1)
    
    fig = go.Figure()

    # Base trace (all points with their original color)
    for (col_val, mod), grp in dff.groupby([color_level, 'modality']):
        color = arch_color_map.get(col_val) if color_level == 'architecture' else fold_color_map.get(col_val)
        fig.add_trace(go.Scattergl(
            x=grp['x'], y=grp['y'], mode='markers',
            name=f"{col_val}, {mod}",
            marker=dict(
                size=grp['size'],
                opacity=grp['opacity'],
                symbol=marker_symbols[mod],
                color=color,
                line=dict(width=0)
            ),
            hovertext=grp['domain_name'],
            customdata=grp[['domain_name', 'domain_id', 'architecture', 'fold', 'x_name', 'h_name', 't_name', 'f_name']],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>ID: %{customdata[1]}<br>Architecture: %{customdata[2]}"
                "<br>Fold: %{customdata[3]}<br>Fold Name: %{customdata[4]}"
                "<br>H Name: %{customdata[5]}<br>T name: %{customdata[6]}<br>F name: %{customdata[7]}<extra></extra>"
            )
        ))

    # Foreground trace: same color, but with black border for matched
    matched = dff[mask]
    if not matched.empty:
        for (col_val, mod), grp in matched.groupby([color_level, 'modality']):
            color = arch_color_map.get(col_val) if color_level == 'architecture' else fold_color_map.get(col_val)
            fig.add_trace(go.Scattergl(
                x=grp['x'], y=grp['y'], mode='markers',
                name='Matched',
                marker=dict(
                    color=color,
                    size=grp['size'],
                    opacity=1.0,
                    symbol=marker_symbols[mod],
                    line=dict(color='black', width=2)
                ),
                hoverinfo='skip',
                showlegend=False
            ))

        # Auto-zoom to matched points
        margin = 5
        x_range = [matched['x'].min() - margin, matched['x'].max() + margin]
        y_range = [matched['y'].min() - margin, matched['y'].max() + margin]
        fig.update_layout(xaxis_range=x_range, yaxis_range=y_range)

    fig.update_layout(
        title="t-SNE ECOD Domains by Modality",
        hovermode='closest',
        margin=dict(l=20, r=20, t=50, b=20),
        dragmode='pan',
        autosize=True
    )

    match_text = f"Highlighted {match_count} result(s)." if search_query else ""
    return fig, match_text


# === App Initialization ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container(
    fluid=True,
    className='vh-100 d-flex flex-column p-3 gap-3',
    children=[
        dbc.Row(dbc.Col(html.H2("ECOD t-SNE Explorer", className='text-center text-primary mb-4'))),
        dbc.Row(
            className='flex-grow-1',
            children=[
                dbc.Col(
                    dbc.Card(
                        className='h-100 d-flex flex-column shadow rounded overflow-hidden',
                        children=[
                            dbc.CardHeader(html.H5("Filters & Search", className='text-center text-white m-0'), className='bg-primary'),
                            dbc.CardBody(
                                className='p-3 d-flex flex-column gap-3 overflow-hidden',
                                children=[
                                    dbc.Label("Color by:"),
                                    dcc.Dropdown(
                                        id='color-level-dropdown',
                                        options=[{'label': 'Architecture', 'value': 'architecture'}, {'label': 'Fold', 'value': 'fold'}],
                                        value='architecture', clearable=False
                                    ),
                                    dbc.Label("Modalities:"),
                                    dcc.Checklist(
                                        id='modality-checklist',
                                        options=[{'label': m.capitalize(), 'value': m} for m in ['sequence','stretch','structure']],
                                        value=['sequence','stretch','structure'],
                                        className='form-check'
                                    ),
                                    dbc.Label("Architecture Filter:"),
                                    dcc.Dropdown(id='architecture-filter-dropdown', multi=True, options=[{'label': arch, 'value': arch} for arch in sorted(df['architecture'].dropna().unique())]),
                                    dbc.Label("Fold Filter:"),
                                    dcc.Dropdown(id='fold-filter-dropdown', multi=True, options=[{'label': f, 'value': f} for f in sorted(df['fold'].dropna().unique())]),
                                    dbc.Label("Search domains:"),
                                    dcc.Input(id='search-input', type='text', debounce=True, className='form-control border-primary fw-bold shadow-sm', placeholder='🔍 Search domains (e.g. kinase)'),
                                    html.Div(id='match-count-text', className='text-center text-secondary fw-bold text-info border rounded py-1 px-2 bg-light shadow-sm'),
                                    html.Hr(),
                                    html.H6("Selected Domains", className='text-center bg-secondary text-white p-2 m-0'),
                                    html.Div(id='selected-info', className='bg-light p-2 rounded overflow-auto flex-grow-1', style={'whiteSpace': 'pre-wrap', 'maxHeight': '300px'}),
                                    dbc.Button("Reset Selection", id='reset-selection', color='secondary')
                                ]
                            )
                        ]
                    ), width=3
                ),
                dbc.Col(
                    dcc.Graph(id='tsne-plot', config={'scrollZoom': True}, style={'height': '100%', 'width': '100%'}, className='h-100'), width=9
                )
            ]
        )
    ]
)

@app.callback(
    [Output('tsne-plot', 'figure'), Output('match-count-text', 'children'), Output('fold-filter-dropdown', 'options')],
    [
        Input('modality-checklist', 'value'),
        Input('search-input', 'value'),
        Input('color-level-dropdown', 'value'),
        Input('architecture-filter-dropdown', 'value'),
        Input('fold-filter-dropdown', 'value'),
    ]
)
def update_plot(selected_modalities, search_query, color_level, architecture_filter, fold_filter):
    if architecture_filter:
        fold_options = [{'label': f, 'value': f} for f, a in fold_arch_map.items() if a in architecture_filter]
    else:
        fold_options = [{'label': f, 'value': f} for f in sorted(df['fold'].dropna().unique())]

    return create_plot(selected_modalities, search_query, color_level, architecture_filter, fold_filter) + (fold_options,)

@app.callback(
    Output('selected-info', 'children'),
    [Input('tsne-plot', 'clickData'), Input('reset-selection', 'n_clicks')],
    [State('selected-info', 'children')]
)
def update_selection(click_data, reset_clicks, current_text):
    triggered = ctx.triggered_id
    if triggered == 'reset-selection':
        return ''
    if click_data and 'customdata' in click_data['points'][0]:
        name, did, arch, fold, x_name, h_name, f_name, t_name = click_data['points'][0]['customdata']
        domain_info = {
            'name': name,
            'id': did,
            'architecture': arch,
            'fold': fold,
            'x_name': x_name,
            'h_name': h_name,
            't_name': t_name,
            'f_name': f_name
        }
        prev = json.loads(current_text) if current_text else []
        prev.append(domain_info)
        return json.dumps(prev, indent=2)
    return current_text

if __name__ == '__main__':
    app.run(debug=True)
