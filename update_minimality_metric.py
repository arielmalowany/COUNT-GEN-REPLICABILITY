# Calcular minimalidad con un threshold más fuerte
# Actualiza avg_features_changed, attribute_changes y agrega factual_atts en cada excel

threshold = 0.5
counterfactuals_dir = './Counterfactuals'

pkl_files = sorted([f for f in os.listdir(counterfactuals_dir) if f.startswith('Front_') and f.endswith('_1.pkl')])

minimality_results = []

for pkl_file in pkl_files:
    front_name = pkl_file.replace('Front_', '').replace('.pkl', '')  # e.g. '22252_0'
    xlsx_path = os.path.join(counterfactuals_dir, f'{front_name}.xlsx')

    with open(os.path.join(counterfactuals_dir, pkl_file), 'rb') as f:
        pareto_front = pkl.load(f)
        fi           = pkl.load(f)
        factual_atts = pkl.load(f)
        _            = pkl.load(f)    # runtime_in_seconds
        _            = pkl.load(f)    # experiment_metadata

    raw_x_data, raw_y_data, raw_z_data, new_preds, new_attributes, generated_cfs, dominance_ranking, crowding_distances = unpack_front(pareto_front)

    valid_cfs_idx = [i for i, y in enumerate(raw_y_data) if y < 0.5]

    if len(valid_cfs_idx) == 0:
        minimality_results.append({'front': front_name, 'avg_features_changed': None, 'count_valid_cfs': 0})
        continue

    valid_cf_atts  = torch.stack([new_attributes[i] for i in valid_cfs_idx])
    count_valid_cfs = len(valid_cfs_idx)

    features_changed     = torch.abs(valid_cf_atts - factual_atts) > threshold
    avg_features_changed = torch.sum(torch.sum(features_changed, axis=1)) / (features_changed.shape[0] * features_changed.shape[1])
    attribute_changes    = torch.sum(features_changed, axis=0) / count_valid_cfs

    factual_atts_str = str(factual_atts.squeeze())

    minimality_results.append({'front': front_name, 'avg_features_changed': round(avg_features_changed.item(), 3)})

    if not os.path.exists(xlsx_path):
        print(f"  Skipping xlsx update — file not found: {xlsx_path}")
        continue

    df = pd.read_excel(xlsx_path)

    # Update avg_features_changed
    mask = df['Metric_Name'] == 'avg_features_changed'
    if mask.any():
        df.loc[mask, 'Value'] = round(avg_features_changed.item(), 3)

    # Update attribute_changes (per-feature boolean counts)
    mask = df['Metric_Name'] == 'attribute_changes'
    if mask.any():
        df.loc[mask, 'Value'] = str(attribute_changes)

    # Add or update factual_atts row
    mask = df['Metric_Name'] == 'factual_atts'
    if mask.any():
        df.loc[mask, 'Value'] = factual_atts_str
    else:
        new_row = pd.DataFrame([[
            'Metadata', 'factual', 'factual_atts', factual_atts_str,
            'Factual image attribute vector'
        ]], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_excel(xlsx_path, index=False)
    print(f"Updated {xlsx_path}")

minimality_df = pd.DataFrame(minimality_results)
minimality_df = minimality_df.melt(id_vars=['front'], value_vars=['avg_features_changed'], var_name='Metric_Name', value_name='Metric_Value')
