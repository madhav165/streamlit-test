import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GoogleSheets:

    # def get_authenticated_service():
    #     credential_path = os.path.join("./", 'sheets.googleapis.com-python-quickstart.json')
    #     if os.path.exists(credential_path):
    #         with open(credential_path, 'r') as f:
    #             credential_params = json.load(f)
    #         credentials = google.oauth2.credentials.Credentials(
    #             credential_params["access_token"],
    #             refresh_token=credential_params["refresh_token"],
    #             token_uri=credential_params["token_uri"],
    #             client_id=credential_params["client_id"],
    #             client_secret=credential_params["client_secret"]
    #         )
    #     else:
    #         flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    #         credentials = flow.run_console()
    #         with open(credential_path, 'w') as f:
    #             p = copy.deepcopy(vars(credentials))
    #             del p["expiry"]
    #             json.dump(p, f, indent=4)
    #     return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

    # def spreadsheets_get(service):
    #     spreadsheetId = "### spreadsheet ID ###"
    #     rangeName = "Sheet1!a1:a10"
    #     results = service.spreadsheets().get(
    #         spreadsheetId=spreadsheetId,
    #         ranges=rangeName
    #     ).execute()
    #     pp.pprint(results)


    # if __name__ == '__main__':
    #     os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    #     service = get_authenticated_service()
    #     spreadsheets_get(service)

    def __init__(self):
        # If modifying these scopes, delete the file token.json.
        self.SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        self.creds = None
        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                if os.path.isfile('token.json'):
                    os.remove('token.json')
                self.creds.refresh(Request())
            else:
                # 
                #     os.remove('token.json')
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(self.creds.to_json())


    def get_sheet_data(self, sheet_id, range_name):
        try:
            service = build('sheets', 'v4', credentials=self.creds)

            # Call the Sheets API
            sheet = service.spreadsheets()
            result = sheet.values().get(spreadsheetId=sheet_id,
                                        range=range_name).execute()
            values = result.get('values', [])

            if not values:
                print('No data found.')
                return

            return values

        except HttpError as err:
            print(err)
            return
