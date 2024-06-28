import json, os
import networkx as nx
from udao_trace.utils import ParquetHandler

def draw_dep(data, title):
    data_dict = json.loads(data)

    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for operator_id, details in data_dict['operators'].items():
        G.add_node(operator_id, label=details['className'].split('.')[-1])  # Use class name as label

    # Add edges
    for link in data_dict['links']:
        G.add_edge(str(link['fromId']), str(link['toId'])) #, label=link['linkType'])

    
    ### Step 3: Visualize the Graph
    
    p = nx.drawing.nx_pydot.to_pydot(G)
    
    dir_to_save = 'figures/logical_query_plans'
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    fig_path = dir_to_save + '/' + title + '.png'
    p.write_png(fig_path)


def generate_logical_query_plan(query_dataset):
    """Plot the logical query plan of each query template in the dataset.

    This function iterates over the query dataset and pplots the logical query plans.
    Args:
        query_dataset (pandas.DataFrame): DataFrame containing the queries.
        
    """
    number_of_seen_templates = 0
    list_of_seen_templates = []
    query_index = 0
    total_number_of_templates = len(query_dataset["template"].unique())
    while number_of_seen_templates < total_number_of_templates:
        template_id = query_dataset["template"].iloc[query_index]
        if template_id in list_of_seen_templates:
            query_index += 1
            continue
        logical_query_plan = query_dataset["lqp"].iloc[query_index]
        title = f"template_{template_id}"
        draw_dep(logical_query_plan, title)
        query_index += 1
        number_of_seen_templates += 1
        list_of_seen_templates.append(template_id)
    print("Finished plotting all queries.")
    

if __name__ == "__main__":
    df = ParquetHandler.load("cache_and_ckp/tpcds_102x490", "df_q_compile.parquet")
    generate_logical_query_plan(df)