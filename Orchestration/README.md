Instructions for hosting Prefect Orion

to set prefect for the server: prefect config set PREFECT_ORION_UI_API_URL="http://34.78.226.69:4200/api"

after setting the above commnad in the terminal, open browser terminal and type"(externalIP):4200"


Steps: 

to start orion instance:

1. prefect orion start --host 0.0.0.0

to run code with prefect: 

1. prefect deployment create prefect_deploy.py
