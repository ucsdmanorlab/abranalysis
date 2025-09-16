from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors


def calculate_unsupervised_threshold(df, freq): # UNUSED
    db_column = db_column_name()

    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return

    waves_array = []  # Array to store all waves

    khz = df[(df['Freq(Hz)'] == freq)]
    db_values = sorted(khz[db_column].unique())
    for db in db_values:
        khz = df[(df['Freq(Hz)'] == freq) & (df[db_column] == db)]

        if not khz.empty:
            index = khz.index.values[-1]
            final = df.loc[index, '0':].dropna()
            final = pd.to_numeric(final, errors='coerce')

            final = final * multiply_y_factor
            if st.session_state.units == 'Nanovolts':
                final /= 1000

            final = interpolate_and_smooth(final)

            y_values = final

            waves_array.append(y_values.tolist())
    # Filter waves and dB values for the specified frequency
    waves_fd = FDataGrid(waves_array)
    fpca_discretized = FPCA(n_components=2)
    fpca_discretized.fit(waves_fd)
    projection = fpca_discretized.transform(waves_fd)

    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    neighbors = nearest_neighbors.fit(projection[:, :2])
    distances, indices = neighbors.kneighbors(projection[:, :2])
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    knee_locator = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    eps = distances[knee_locator.knee]

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps)
    clusters = dbscan.fit_predict(projection[:, :2])

    # Create DataFrame with projection results and cluster labels
    dfn = pd.DataFrame(projection[:, :2], columns=['1st_PC', '2nd_PC'])
    dfn['Cluster'] = clusters
    dfn['DB_Value'] = db_values

    # Find the minimum hearing threshold value among the outliers
    min_threshold = np.min(dfn[dfn['Cluster']==-1]['DB_Value'])

    return min_threshold
