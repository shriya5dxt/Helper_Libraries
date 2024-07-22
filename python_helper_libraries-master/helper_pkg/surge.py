import pandas as pd
# from tqdm import tqdm
from . import Helper
import re
import numpy as np
from tqdm import tqdm
tqdm.pandas()


class SURGE(Helper):
    def __init__(self):
        self.xyz = 'xyz'
        Helper.__init__(self)

    def get_part_order_details(self, table, se_pod_col='part_order_identifier', pod_col='part_order_identifier1', pod_stg_table='STG_CDAX_PARTORDERDETAIL_ATTR_2020_10_08'):

        pods = self.get_missing_pods(table, se_pod_col, pod_col)

        pod_df = self.get_pod_info(pods)

        drop_old_pod_stg = f"""drop table pps_quality.{pod_stg_table};"""
        self.run_query(drop_old_pod_stg)

        status = self.load_to_vertica_database(
            pod_df, self.schema, pod_stg_table)

        if status == 'Successful':
            update_status = self.update_stg_to_master_pod(
                table, pod_col, pod_stg_table, se_pod_col)

            if update_status:
                print('Completed successfully!')
                return True
            else:
                print('Failed in updating part_order_details')
                return False
        else:
            print(
                f'Failed load_to_vertica_database while loading to {pod_stg_table}')
            return False

    def get_missing_pods(self, table, se_pod_col='part_order_identifier', pod_col='part_order_identifier1'):
        missing_pods_query = f"""select distinct {se_pod_col}
                            from pps_quality.{table}
                            where {pod_col} is null
                            and {se_pod_col} is not null
                            ;
                            """
        with self.get_connection() as aws_conn:
            missing_pods = pd.read_sql(missing_pods_query, aws_conn)

        return missing_pods[se_pod_col].to_list()
    
    def convert_uuid_to_str(df, col):
        df[col] = df[col].astype(str)
        df[col] = df[col].replace('nan', np.nan)

    def get_pod_info(self, pods):
        chunks = self.divide_into_chunks(pods, 10000)
        stg_dataframes = []
#     df = pd.DataFrame()
        for chunk in tqdm(chunks):
            maincase_ids_str = "', '".join(chunk)
    
            extract_pod_info_query = f"""Select BINARY_CHECKSUM(cb.Part_Order_identifier, 
                        cb.offered_part_number,cb.offered_part_description,
                        cc.failure_description,concat(cb.Ship_Plant , cb.Ship_From_Storage_Location_Name),
                        substring(USAGE_CODE.LOCALIZED_LABEL_NAME,6,21),PODSS.Localized_Label_Name) as ID, 
                        cb.Part_Order_identifier, 
                        cb.offered_part_number as PO_Part_Num, 
                        cb.offered_part_description as PO_Part_Desc,
                        cc.failure_description as failure_desc, 
                        concat(cb.Ship_Plant,cb.Ship_From_Storage_Location_Name) as Source_Location, 
                        substring(USAGE_CODE.LOCALIZED_LABEL_NAME,6,21) as Parts_Usage, 
                        PODSS.Localized_Label_Name as Detailed_PO_Status,
                        cb.Update_GMT_Timestamp 
                        from Common_css_gcsi.Fact_Service_Part_Order_Detail_view as cb 
                        left join Common_css_gcsi.Dim_Service_Failure_view as cc 
                        on cc.failure_identifier = cb.failure_Identifier
                        LEFT JOIN Common_css_gcsi.DIM_SERVICE_OPTION_SET_METADATA_VIEW USAGE_CODE 
                        ON (USAGE_CODE.Entity_Name = 'hpi_partorderdetail'
                        AND USAGE_CODE.OPTION_SET_NAME = 'hpi_podpartusagecode'
                        AND cb.Pod_Part_Usage_Code = USAGE_CODE.OPTION_NUMBER
                        AND USAGE_CODE.LOCALIZED_LABEL_LANGUAGE_CODE = 1033
                        AND USAGE_CODE.IS_USER_LOCALIZED_LABEL_FLAG = 0) 
                        left join  Common_css_gcsi.Dim_Service_State_Metadata_View as PODS
                        ON (cb.Entity_State_Code = PODS.Entity_State_Code 
                        and PODS.Entity_Name = 'hpi_partorderdetail' 
                        and PODS.Localized_Label_Language_Code = 1033 
                        and PODS.Is_User_Localized_Label_Flag = 0)
                        left Join Common_css_gcsi.Dim_Service_Status_Metadata_View as PODSS
                        ON (cb.Entity_Status_Code = PODSS.Entity_Status_Code 
                        and PODSS.Entity_Name = 'hpi_partorderdetail' 
                        and PODSS.Localized_Label_Language_Code = 1033 
                        and PODSS.Is_User_Localized_Label_Flag = 0)
                where
                part_order_identifier in ('{maincase_ids_str}')
                --DATEDIFF(MONTH,cb.Update_GMT_Timestamp , GETDATE()) <= 6 
                and cb.offered_part_number IS NOT NULL 
                and PODSS.Localized_Label_Name NOT LIKE 'Cancel%' 
                and PODSS.Localized_Label_Name NOT LIKE 'Inactive%' 
                and cb.Part_Order_Identifier IS NOT NULL
                
                ;"""
       


            counter = 0
            total_tries = 5
            while(counter < total_tries):
                try:
                    with self.get_connection('sql_cache_surge') as cdax_conn:
                        print('Connected to sql_cache_surge')
                        df = pd.read_sql(extract_pod_info_query, cdax_conn)
                        self.convert_uuid_to_str(df, 'ID')
                        self.convert_uuid_to_str(df, 'Part_Order_identifier')
                        self.convert_uuid_to_str(df, 'PO_Part_Num')
                        self.convert_uuid_to_str(df, 'PO_Part_Desc')
                        self.convert_uuid_to_str(df, 'failure_desc')
                        self.convert_uuid_to_str(df, 'Source_Location')
                        self.convert_uuid_to_str(df, 'Parts_Usage')
                        self.convert_uuid_to_str(df, 'Detailed_PO_Status')
                        self.analyse_df(df)
                        break
                except Exception as e:
                    print(f'Failed to connect to sql_cache: attempt {counter+1}')
                    counter += 1
                    if counter < total_tries:
                        continue
                    else:
                        print('Max retries reached. Breaking out of the loop.')
                        # self.send_mail()
                        break

            stg_dataframes.append(df)
        stg = pd.concat(stg_dataframes, ignore_index=True, sort=False)
        self.analyse_df(stg)
    
        if len(df) == 0:
            return 'No update available'

    

        return df

    def update_stg_to_master_pod(self, table, pod_col, pod_stg_table, se_pod_col):
        update_stg_to_master = f"""UPDATE pps_quality.{table} as p
                    SET {pod_col} = s.Part_Order_identifier,
                        po_part_num = s.po_part_num,
                        failure_desc = s.failure_desc,
                        source_location = s.source_location,
                        parts_usage = s.parts_usage,
                        detailed_po_status = s.detailed_po_status,
                        id__pps_qms_surge_partorderdetail_attr_ = s.id
                    FROM PPS_QUALITY.{pod_stg_table} as s
                    WHERE p.{se_pod_col} = s.Part_Order_Identifier;"""
        update_status = self.run_query(update_stg_to_master, join_spill=True)
        return update_status
