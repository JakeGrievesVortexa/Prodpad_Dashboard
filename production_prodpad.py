import pandas as pd
import streamlit as st
import glob
import os
from datetime import datetime
import re

st.markdown("<h1 style='text-align: left;'>ProdPad Use</h1>", unsafe_allow_html=True)

folder_path = '/Users/jakegrieves/Desktop/Python Scripts/Jupyter Notebooks/prodpad_data/'
workflow_order = ['Unsorted','New Idea','Discovery','Business Clarify','R&D Clarify','Ready for Prioritisation','Prioritised','On JIRA','Released','Not Doing']
active_workflow_order = ['Discovery','Business Clarify','R&D Clarify','Ready for Prioritisation','Prioritised','On JIRA','Released']

def update_fields(df):
    df["owner"] = df["owner"].apply(lambda x: "None" if pd.isna(x) else eval(x)['display_name'])
    df['status_updated'] = df.apply(lambda x: x['created_at'] if pd.isna(x['status']) else eval(x['status'])['added'],axis = 1)
    df['status'] = df['status'].apply(lambda x: None if pd.isna(x) else eval(x)['status']) 
    df['status_updated'] = pd.to_datetime(df['status_updated']).dt.date
    df.loc[(df['status'].isna()) & (df['state'] == 'active'), 'status'] = 'New Idea'
    df.loc[(df['status'].isna()) & (df['state'] == 'unsorted'), 'status'] = 'Unsorted'
    df['updated_at'] = pd.to_datetime(df['updated_at']).dt.date
    df['created_at'] = pd.to_datetime(df['created_at']).dt.date
    df['downloaded_at'] = pd.to_datetime(df['downloaded_at']).dt.date
    return df

def get_csv_files():
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    csv_files = [re.search(r'\d{4}-\d{2}-\d{2}', file).group() for file in csv_files]
    csv_files.sort(reverse=True)
    return(csv_files)

def create_header(df,prev_df):
    current_val = pd.DataFrame(df['status'].value_counts()) 
    prev_val = pd.DataFrame(prev_df['status'].value_counts())
    current_val = current_val.rename(columns={'status':'current'})
    current_val["previous"] = prev_val['count']

    current_val = current_val.reindex(workflow_order)
    prev_val = prev_val.reindex(workflow_order)
    current_val.fillna(0, inplace=True)
    current_val['previous'] = current_val['previous'].astype(int)
    current_val['diff'] = current_val['count'] - current_val['previous']

    col = st.columns(10)
    for i,(index,row) in enumerate(current_val.iterrows()):
        with col[i]:
            st.metric(label = index, value = int(row['count']),delta = no_change(int(row['diff'])))

    return()

def no_change(integer):
    if integer == 0:
        return None
    else:
        return integer

def get_status_changes(df,prev_df):
    # DF of ideas that have changed status
    st.title("Ideas That Have Changed Status:")
    merged = pd.merge(df, prev_df, how='left', on='project_id', suffixes=('_current', '_old'))
    st.write(merged[["project_id","title_current","owner_current","description_current","status_old","status_current","state_old","state_current"]].loc[merged["status_current"] != merged["status_old"]])
    # DF of ideas that have been archived
    st.title("Archived Ideas:")
    merged = pd.merge(df, prev_df, how='right', on='project_id', suffixes=('_current', '_old'))
    st.write(merged[["project_id","title_old","owner_old","description_old","status_old","state_old"]].loc[merged["status_current"].isna()])

def product_owner_ideas(df,prev_df):
    st.title("Product Manager Ideas")


    project_owner_filter = st.selectbox(
        'Select Product Manager',
        options=df['owner'].unique()
    )

    current_val = pd.DataFrame(df['status'][df["owner"] == project_owner_filter].value_counts())
    prev_val = pd.DataFrame(prev_df['status'][prev_df["owner"] == project_owner_filter].value_counts())
    current_val = current_val.rename(columns={'status':'current'})
    current_val["previous"] = prev_val['count']

    current_val = current_val.reindex(workflow_order)
    prev_val = prev_val.reindex(workflow_order)
    current_val.fillna(int(0), inplace=True)
    current_val['previous'] = current_val['previous'].astype(int)
    current_val['diff'] = current_val['count'] - current_val['previous']

    col = st.columns(len(workflow_order))
    for i,(index,row) in enumerate(current_val.iterrows()):
        with col[i]:
            st.metric(label = index, value = int(row['count']),delta = no_change(int(row['diff'])))
    
    idea_status_select = st.selectbox(
    'Select Status',
    options=df['status'].unique()
    )

    st.write(df[['project_id','title','description','owner','status','updated_at']].loc[(df["owner"] == project_owner_filter) & (df["status"] == idea_status_select)])


def idea_lookup(csv_files):
    st.title("Track Ideas")

    idea_select = st.number_input(
        'Track Idea:',
        min_value=0
    ) 
    dataframes = []
    for i in csv_files:
        csv_df = pd.read_csv(f'{folder_path}{i}.csv')
        try:
            dataframes.append(csv_df.loc[csv_df['project_id'] == idea_select])
        except:
            None



    combined_df = pd.concat(dataframes, ignore_index=True)

    if len(combined_df) == 0:
        st.error("Invalid Id")
    else:
        st.write(combined_df)

        def compare_rows(row1, row2):
            changes = []
            for col in row1.index:
                if row1[col] != row2[col]:
                    changes.append(f"{col} Change: {row1[col]} -> {row2[col]}")
            return changes

        # Iterate through the DataFrame and compare each row with the previous one
        def track_changes(df):
            changes = []
            for i in range(1, len(df)):
                row1 = df.iloc[i-1]
                row2 = df.iloc[i]
                change = compare_rows(row1, row2)
                if change:
                    changes.append(f"{row2['updated_at']}: " + ", ".join(change))
            return changes

        combined_df["owner"] = combined_df["owner"].apply(lambda x: "None" if pd.isna(x) else eval(x)['display_name'])
        combined_df['status'] = combined_df['status'].apply(lambda x: 'Unsorted' if pd.isna(x) else eval(x)['status'])
        combined_df['updated_at'] = pd.to_datetime(combined_df['updated_at']).dt.date
        combined_df['created_at'] = pd.to_datetime(combined_df['created_at']).dt.date
        changes = track_changes(combined_df[["updated_at","owner","status"]])

        # Display changes
        st.write("Changes in the project:")
        st.write(f"""Created: {combined_df["created_at"].iloc[0]}""")
        for change in changes:
            st.write(change)

def cycle_times(csv_files):
    st.subheader('Average Cycle Times')
    csv_dataframe = []
    for i in csv_files:
        csv_dataframe.append(pd.read_csv(f'{folder_path}{i}.csv'))

    cycle_df = pd.concat(csv_dataframe)
    cycle_df = update_fields(cycle_df)

    cycle_times = {}
    for status in cycle_df['status'].unique():
        dataframe = []
        for project_id in cycle_df['project_id'].loc[cycle_df['status'] == status].unique():
            time = cycle_df['downloaded_at'].loc[(cycle_df['project_id'] == project_id) & (cycle_df['status'] == status)].max() - cycle_df['downloaded_at'].loc[(cycle_df['project_id'] == project_id) & (cycle_df['status'] == status)].min()
            dataframe.append(time)
        avg=pd.to_timedelta(pd.Series(dataframe).mean())
        cycle_times[status] = avg

    cycles = st.columns(len(workflow_order))

    for index,status in enumerate(workflow_order):
        with cycles[index]:
            st.metric(label = status, value = str(cycle_times[status].days) + " Days")

    st.subheader('Average Cycle Time Per Product Manager')

    cycle_owner_filter = st.selectbox(
        'Select Product Manager',
        options=cycle_df['owner'].unique(),
        key = 'cycle_manager'
    )

    prod_man_cycle_times = {}
    for status in cycle_df['status'].unique():
        dataframe = []
        for project_id in cycle_df['project_id'].loc[(cycle_df['status'] == status) & (cycle_df['owner'] == cycle_owner_filter)].unique():
            time = cycle_df['downloaded_at'].loc[(cycle_df['project_id'] == project_id) & (cycle_df['status'] == status) & (cycle_df['owner'] == cycle_owner_filter)].max() - cycle_df['downloaded_at'].loc[(cycle_df['project_id'] == project_id) & (cycle_df['status'] == status) & (cycle_df['owner'] == cycle_owner_filter)].min()
            dataframe.append(time)
        avg=pd.to_timedelta(pd.Series(dataframe).mean())
        prod_man_cycle_times[status] = avg

    cycles = st.columns(len(workflow_order))

    for index,status in enumerate(workflow_order):
        with cycles[index]:
            if  type(prod_man_cycle_times[status].days) != int:
                st.metric(label = status, value = "NA")
            else:
                st.metric(label = status, value = str(prod_man_cycle_times[status].days) + " Days",delta = prod_man_cycle_times[status].days - cycle_times[status].days,delta_color='inverse')

def cycle_times_using_update_status(df,csv_files):
    st.subheader('Average Cycle Times (Days)')
    csv_dataframe = []
    for i in csv_files:
        csv_dataframe.append(pd.read_csv(f'{folder_path}{i}.csv'))

    cycle_df = pd.concat(csv_dataframe)
    cycle_df = update_fields(cycle_df)

    keys = {}
    for project_id in df['project_id'].loc[df['state'] != "unsorted"].unique():
        keys[project_id] = json_output(cycle_df,project_id)

    status_avg_time = {}

    for project_id, status_dict in keys.items():
        for status, status_info in status_dict.items():
            if status not in status_avg_time:
                status_avg_time[status] = []
            status_avg_time[status].append(status_info['time_in_status'])

    avg_time_in_status = {}

    for status, time_list in status_avg_time.items():
        avg_time_in_status[status] = sum(time_list) / len(time_list)

    cycles = st.columns(len(active_workflow_order))

    for index,status in enumerate(active_workflow_order):
        with cycles[index]:
            try: 
                st.metric(label = status, value = str(round(avg_time_in_status[status],1)))
            except:
                st.metric(label = status, value = "NA")

    st.subheader('Average Cycle Time Per Product Manager')

    project_owner_filter = st.selectbox(
        'Select Product Manager',
        options=cycle_df['owner'].unique(),
        key = 'cycle_manager'
    )

    manager_keys = {}
    for project_id in df['project_id'].loc[(df['owner'] == project_owner_filter) & (df['state'] != "unsorted")].unique():
        manager_keys[project_id] = keys[project_id]

    status_avg_time = {}

    for project_id, status_dict in manager_keys.items():
        for status, status_info in status_dict.items():
            if status not in status_avg_time:
                status_avg_time[status] = []
            status_avg_time[status].append(status_info['time_in_status'])

    managers_time_in_status = {}

    for status, time_list in status_avg_time.items():
        managers_time_in_status[status] = sum(time_list) / len(time_list)

    cycles = st.columns(len(active_workflow_order))
    #st.write(managers_time_in_status)
    for index,status in enumerate(active_workflow_order):
        with cycles[index]:
            try:
                st.metric(label = status, value = str(round(managers_time_in_status[status],1)),delta = round(managers_time_in_status[status] - avg_time_in_status[status],1),delta_color='inverse')
            except:
                st.metric(label = status, value = "NA")






def time_in_status(df,csv_files):
    st.title('Ideas Stuck in a Status')

    # First Iterate through the df and get the date downloaded, status and project id.
    # then I want to iterate through the combined_df and find the time it has been at that status.

    # user_defined_days = st.number_input(
    #     'Number of days to check:',
    #     min_value=1
    # )
    csv_dataframe = []
    for i in csv_files:
        csv_dataframe.append(pd.read_csv(f'{folder_path}{i}.csv'))

    cycle_df = pd.concat(csv_dataframe)
    cycle_df = update_fields(cycle_df)

    def add_time_at_status(row):
        return(cycle_df['downloaded_at'].loc[(cycle_df['project_id'] == row['project_id']) & (cycle_df['status'] == row['status'])].max() - cycle_df['downloaded_at'].loc[(cycle_df['project_id'] == row['project_id']) & (cycle_df['status'] == row['status'])].min())

    df['time_at_status'] = df.apply(lambda x: add_time_at_status(x), axis=1)

    st.write(df[['project_id','title','owner','status','time_at_status']].sort_values(by = 'time_at_status',ascending = False))

def current_time_in_status(df,csv_files):
    df = df.loc[df['state'] != "unsorted"]
    st.subheader('Ideas Stuck in a Status')
    csv_dataframe = []
    for i in csv_files:
        csv_dataframe.append(pd.read_csv(f'{folder_path}{i}.csv'))

    cycle_df = pd.concat(csv_dataframe)
    cycle_df = update_fields(cycle_df)

    keys = {}
    for project_id in df['project_id'].unique():
        keys[project_id] = json_output(cycle_df,project_id)

    def add_time_at_status(row):
        return(keys[row['project_id']][row['status']]['time_in_status'])

    df['time_at_status'] = df.apply(lambda x: add_time_at_status(x), axis=1)

    st.write(df[['project_id','title','owner','status','time_at_status']].loc[df['status'] != "New Idea"].sort_values(by = 'time_at_status',ascending = False))
#JSON? 


# This is a function that given a single idea will return a json of the time spent in each status:
# including start and end date:
def json_output(cycle_df,idea_id):
    # if there is only one status then we can return the time spent in that status
    # and the start date will be the first time the status was updated
    # this was originally the created_at date however it runs into weird issues due to us having no backlog,
    # i.e. a project is currently in priority but we have no backlog for it, it was created in 2020, 
    # it looks like the project has been there for 4 years which is incorrect
    # the end date will be the current date

    if len(cycle_df['status'].loc[cycle_df['project_id'] == idea_id].unique()) <= 1:
        return(
        {
            cycle_df['status'].loc[cycle_df['project_id'] == idea_id].unique()[0]: 
            {
                "started_at": cycle_df['status_updated'].loc[cycle_df['project_id'] == idea_id].min(),
                "ended_at": datetime.now().date(),
                "time_in_status": (datetime.now().date() - cycle_df['status_updated'].loc[cycle_df['project_id'] == idea_id].min()).days
            }
        }    
        )
    # if there are multiple statuses then we need to iterate through the statuses and find the time spent in each status
    # we move thorugh the DF in order of the downloaded_at date ascending to the current date
    # the start date will be the first time the status was updated
    # the end date will be the first time the next status was updated
    # if there is no next status then the end date will be the current date

    else:
        counter = []
        output = {}
        for index,row in cycle_df.loc[cycle_df['project_id'] == idea_id].sort_values('downloaded_at',ascending = True).iterrows():
            if index == 0:
                output[row['status']] = {
                    "started_at": row['created_at'],
                    "ended_at": None,
                    "time_in_status": None
                }
            elif row['status'] not in counter:
                output[row['status']] = {
                    "started_at": row['status_updated'],
                    "ended_at": None,
                    "time_in_status": None
                }
    
                if len(counter) >= 1:
                    output[counter[-1]]['ended_at'] = row['status_updated']
                    output[counter[-1]]['time_in_status'] = (output[counter[-1]]['ended_at'] - output[counter[-1]]['started_at']).days
                
                counter.append(row['status'])
        if len(counter) >= 1:
            print(counter)
            output[counter[-1]]['ended_at'] = datetime.now().date()
            output[counter[-1]]['time_in_status'] = (output[counter[-1]]['ended_at'] - output[counter[-1]]['started_at']).days
        return output

        

def main():
    #Get CSV files
    csv_files = get_csv_files()

    #User Selects CSV files 
    start_csv = st.selectbox(label = "End Date", options = csv_files)
    end_csv = st.selectbox(label = "Start Date", options = csv_files)

    #Ensure start date is less than end date
    start_date = datetime.strptime(start_csv, '%Y-%m-%d')
    end_date = datetime.strptime(end_csv, '%Y-%m-%d')
    if start_date < end_date:
        st.error("Start date cannot be greater than end date.")

    #Load CSV files
    df = pd.read_csv(f'{folder_path}{start_csv}.csv')
    prev_df = pd.read_csv(f'{folder_path}{end_csv}.csv')

    #Update Fields
    df = update_fields(df)
    prev_df = update_fields(prev_df)

    #Create Header
    create_header(df,prev_df)

    #Ideas with Changes Status
    get_status_changes(df,prev_df)

    #Product Manager Ideas
    product_owner_ideas(df,prev_df)

    #idea lookup
    #idea_lookup(csv_files)

    #Cycle Times
    cycle_times_using_update_status(df,csv_files)

    #Time in Status
    current_time_in_status(df,csv_files)
    #time_in_status(df,csv_files)
    


main()







