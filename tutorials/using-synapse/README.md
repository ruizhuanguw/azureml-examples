# Azure ML & Synapse integration - Private Preview (July 2020)

#### What is Azure Synapse Analytics (formerly SQL DW)?
Azure Synapse is an analytics service that brings together enterprise data warehousing and Big Data analytics. It gives you the freedom to query data on your terms, using either serverless on-demand or provisioned resources—at scale. Azure Synapse brings these two worlds together with a unified experience to ingest, prepare, manage, and serve data for immediate BI and machine learning needs.

#### What are you launching?
We are launching managed Apache Spark (backed by Synapse) for customers to perform interactive and performant data preparation for machine learning in Azure ML. Customers can use Azure ML notebooks to connect to spark clusters to do interactive data preparation using PySpark. Customers have the option to configure spark sessions and quickly experiment and iterate on the data. Once ready, they can leverage Azure ML pipelines to automate their end to end ML workflow from data preparation to model deployment all in one platform. Furthermore, with the seamless integration with Synapse, customers can register their Spark clusters and data-source to Azure ML workspace with just a few clicks. 

#### Who is the feature intended for?
This feature is for customers who want to use Spark to prepare data at scale in Azure ML before training their ML model. This will allow customers to work on their end to end ML lifecycle including large-scale data preparation, model training and deployment within Azure ML workspace without having to use suboptimal tools for machine learning or switch between multiple tools for data preparation and model training. The ability to perform all ML tasks within Azure ML will reduce time required for customers to iterate on a machine learning project which typically includes multiple rounds of data preparation and training.   

#### Private preview scope
Users in private preview will be able to compelete the following tasks via SDK:
1. Link Synapse workspace to Azure ML workspace
2. Attach Apache Spark pools backed by Synapse as Azure ML computes
3. Launch Spark session in notebooks and perform interactive data exploration and preparation
4. Productionize ML pipelines by leveraging spark pools to pre-process big data 
5. Run training and scoring on spark if needed

