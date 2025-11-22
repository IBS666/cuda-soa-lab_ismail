pipeline {
    agent any
    stages {
        stage('Clone Repo') {
            steps {
                git 'https://github.com/IBS666/cuda-soa-lab.git'
            }
        }
        stage('Build Docker') {
            steps {
                sh 'docker build -t gpu-service .'
            }
        }
        stage('Deploy') {
            steps {
                sh 'docker run --gpus all -d -p 8000:8000 gpu-service'
            }
        }
    }
}
