## 5.1 Intro / Session overview

In this session, we talked about the earlier model we made in chapter 3 for churn prediction.
This chapter contains the deployment of the model. If we want to use the model to predict new values without running the code, there's a way to do this. The way to use the model in different machines without running the code, is to deploy the model in a server (run the code and make the model). After deploying the code in a machine used as server we can make some endpoints (using api's) to connect from another machine to the server and predict values.

Model deployment is crucial when you need to use the model across different machines or applications without having to retrain or rerun the code. By deploying the model as a web service, external systems (like marketing services) can send requests to the server to get predictions, such as whether a customer is likely to churn. Based on the prediction, actions like sending promotional offers can be automated.

To deploy the model in a server there are some steps:

Train and Save the Model: After training the model, save it as a file, to use it for making predictions in future (session 02-pickle).
Create API Endpoints: Make the API endpoints in order to request predictions. It is possible to use the Flask framework to create web service API endpoints that other services can interact with (session 03-flask-intro and 04-flask-deployment).
Some other server deployment options (sessions 5 to 9):
Pipenv: Create isolated environments to manage the Python dependencies of the web service, ensuring they don't interfere with other services on the machine.
Docker: Package the service in a Docker container, which includes both system and Python dependencies, making it easier to deploy consistently across different environments.
Deploy to the Cloud: Finally, deploy the Docker container to a cloud service like AWS to make the model accessible globally, ensuring scalability and reliability.