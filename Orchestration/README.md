Instructions for hosting Prefect Orion on VM

to set prefect for the server: prefect config set PREFECT_ORION_UI_API_URL="http://34.78.84.50:4200/api"

to check if IP is registered: prefect config view

tp reset the prefect IP: prefect config unset PREFECT_ORION_UI_API_URL



after setting the above commnad in the terminal, open browser terminal and type"(externalIP):4200"


Steps: 

to start orion instance:

1. prefect orion start --host 0.0.0.0

to run code with prefect: 

1. prefect deployment create prefect_deploy.py
