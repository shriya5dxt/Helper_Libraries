import pandas as pd
import requests

from helper_pkg import Helper

hlp = Helper()


class CDAX_Realtime(Helper):
    client_id = '198abf37-4c34-401f-98c8-c52509041fc3'
    scope = 'https://hpcdax.crm.dynamics.com/user_impersonation offline_access'
    redirect_uri = 'http://localhost:9000'
    client_secret = '/8=0F?/9BJHg3DG1c4=@m=M8M:KMpjv:'
    refresh_url = 'https://login.microsoftonline.com/common/oauth2/v2.0/token'

    cdax_url = 'https://hpcdax.crm.dynamics.com/api/data/v9.1/'

    tokens_table = 'CDAX_TOKENS'
    read_tokens_query = f"""select * from pps_quality.{tokens_table};"""
    with hlp.get_connection('aws3') as conn:
        tokens_df = pd.read_sql(read_tokens_query, conn)

    access_token = tokens_df.loc[0, 'access_token']
    refresh_token = tokens_df.loc[0, 'refresh_token']

    def __init__(self):
        super().__init__()

    def gen_access_token(self):
        print('Generating new access token')

        body = {
            'client_id': CDAX_Realtime.client_id,
            'scope': CDAX_Realtime.scope,
            'redirect_uri': CDAX_Realtime.redirect_uri,
            'grant_type': 'refresh_token',
            'client_secret': CDAX_Realtime.client_secret,
            'refresh_token': CDAX_Realtime.refresh_token
        }
        response = requests.request(
            'POST', CDAX_Realtime.refresh_url, data=body)
        if response.ok:
            print('Access token generated successfully!')
            json_response = response.json()
            CDAX_Realtime.tokens_df.loc[0,
                                        'access_token'] = json_response['access_token']
            CDAX_Realtime.access_token = json_response['access_token']
            CDAX_Realtime.tokens_df.loc[0,
                                        'refresh_token'] = json_response['refresh_token']
            CDAX_Realtime.refresh_token = json_response['refresh_token']

            self.run_query(f"""update pps_quality.{CDAX_Realtime.tokens_table}
                                set access_token = '{json_response['access_token']}',
                                refresh_token = '{json_response['refresh_token']}'
            """)

            return True

        else:
            print('Access token generation failed!')
            json_response = response.json()
            print(f"Error: {json_response['error']}")
            print(f"Error Description: {json_response['error_description']}")
            return False

    def get_data(self, endpoint, access_token, filters=None):
        if filters is None:
            master_url = f'{CDAX_Realtime.cdax_url}{endpoint}'
        else:
            master_url = f'{CDAX_Realtime.cdax_url}{endpoint}?{filters}'

        # print(f'Querying {master_url}')

        payload = ''
        headers = {
            'Authorization': f'Bearer {access_token}',
        }
        response = requests.request(
            "GET", master_url, headers=headers, data=payload)

        if response.ok:
            json_response = response.json()
            df = pd.DataFrame(json_response['value'])
            return df
        else:
            print('Access token expired')
            access_token_generated = self.gen_access_token()
            if access_token_generated:
                headers = {
                    'Authorization': f'Bearer {CDAX_Realtime.access_token}'
                }
                response = requests.request(
                    "GET", master_url, headers=headers, data=payload)
                json_response = response.json()
                if response.ok:
                    df = pd.DataFrame(json_response['value'])
                    return df
                else:
                    print('Failed to get data')
                    print(f"Error: {json_response['error']}")
                    print(
                        f"Error Description: {json_response['error_description']}")
            else:
                print('Access token generation failed')

    def get_data_batch_wise(self, source_endpoint_data, target_endpoint, left_on, right_on, query_params='', batch_size=150):
        """Fetches data from a cdax endpoint for specific ids in batches.

        Args:
            source_endpoint_data (pandas.DataFrame): Endpoint data collected previously from the get_data function
            target_endpoint (str): CDAX entity to get data from.
            left_on (str): Column name of source_endpoint_data dataframe to join on.
            right_on (str): Column name of target entity on which join will occur.
            query_params (str, optional): Any valid odata query parameter. Please ensure $filter is added last else the function will fail. Defaults to ''.
            batch_size (int, optional): Size of batches. Generally fails if >190. Defaults to 150.

        Returns:
            pandas.DataFrame: The required data from target_endpoint for the ids provided in source_endpoint_data
        """
        ids = source_endpoint_data[left_on].to_list()
        # batch_size = 150

        results = []
        start = 0
        end = min(batch_size, len(ids))
        print('Total IDs: ', len(ids))

        if len(query_params) != 0 and '$filter=' not in query_params:
            query_params += '&'

        while start != end:
            batch_ids = ids[start:end]
            batch_ids_url = f' or {right_on} eq '.join(batch_ids)

            if '$filter=' not in query_params:
                batch_ids_url = f'$filter={right_on} eq ' + batch_ids_url
                final_filters = query_params + batch_ids_url
            else:
                batch_ids_url = f' and {right_on} eq ' + batch_ids_url
                final_filters = query_params + batch_ids_url

            print(f'Getting data for {start}:{end} ')

            result = self.get_data(target_endpoint,
                                   filters=final_filters,
                                   access_token=CDAX_Realtime.access_token)

            results.append(result)

            start = min(end, len(ids))
            end = min(start + batch_size, len(ids))

        final_result = pd.concat(results, sort=False, ignore_index=True)

        return final_result

    # def __get_all_case_data_skynet(self, row):

    #     systemuser = self.get_data('systemusers',
    #                                filters=f"$filter=systemuserid eq {row['_ownerid_value']}&$select=systemuserid, yomifullname, windowsliveid",
    #                                access_token=CDAX_Realtime.access_token)
    #     glb.systemusers.append(systemuser)

    #     workorder = self.get_data('msdyn_workorders',
    #                               filters=f"$filter=_msdyn_servicerequest_value eq {row.incidentid}",
    #                               access_token=CDAX_Realtime.access_token)
    #     glb.msdyn_workorders.append(workorder)

    #     if len(workorder) > 0:
    #         workorder = workorder[-1:]
    #         wid = workorder['msdyn_workorderid'].values[0]
    #         booking_info = self.get_data('bookableresourcebookings',
    #                                      filters=f"""$filter=_msdyn_workorder_value eq {wid}""",
    #                                      access_token=CDAX_Realtime.access_token)
    #         glb.bookableresourcebookings.append(booking_info)

    #     return row


if __name__ == "__main__":
    cdax_rlt = CDAX_Realtime()
    test_data = cdax_rlt.get_data(
        'incidents', access_token=CDAX_Realtime.access_token, filters='$top=2&$select=hpi_caseid')
    print(test_data)
