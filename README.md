
# Application Setup Guide

This guide provides instructions on how to install and run the application using Docker Compose on Ubuntu and Windows systems. This setup includes the necessary steps to configure Python, Azure, OpenAI integrations, and environmental variables.

## Prerequisites

Before you begin, ensure that the following tools are installed on your system:

1. **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
2. **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)
3. **Python**: Required for local scripts and testing. [Install Python](https://www.python.org/downloads/)
4. **Azure CLI**: Optional for Azure-specific configurations. [Install Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

## Configuration

### Setting up Environment Variables

Change values in the `.env` file in the `backend` folder of your project directory. This file will store all your application-specific environment variables. Here is an example structure:

```
Azure_API_KEY="your_azure_api_key_here"
OPENAI_API_BASE="https://bt-gpt4-preview.openai.azure.com"
OPENAI_API_VERSION="2024-03-01-preview"
```

Replace `your_openai_api_key_here` with your actual API keys from Azure OpenAI services in section Keys and Endpoint.

## Running the Application

To start the application, navigate to the directory containing your `docker-compose.yml` file and run:

```bash
docker-compose up --build
```

This command builds the image if it does not exist and starts the containers as defined in your Docker Compose configuration.

## Shutting Down

To stop and remove the containers, use the following command:

```bash
docker-compose down
```

## Platform-Specific Notes

### Windows

Ensure that Docker Desktop is configured to use Linux containers. You can switch between Linux and Windows containers from the Docker tray icon.

### Ubuntu

Ensure your user is added to the `docker` group to run Docker commands without `sudo`:

```bash
sudo usermod -aG docker ${USER}
su - ${USER}
```

Log out and back in for this to take effect.

