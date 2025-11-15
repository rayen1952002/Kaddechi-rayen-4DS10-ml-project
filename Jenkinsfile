pipeline {
    agent {
        docker {
            image 'python:3.9-slim'
            args '-u root'
        }
    }
    
    environment {
        MLFLOW_TRACKING_URI = 'file:///var/jenkins_home/mlruns'
    }
    
    stages {
        stage('Test') {
            steps {
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
                sh 'pip install -r requirements-dev.txt'
                sh 'python -m pytest tests/ -v --junitxml=test-results.xml'
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }
        
        stage('Lint') {
            steps {
                sh 'pip install black flake8'
                sh 'black --check src/ tests/'
                sh 'flake8 src/ tests/ --max-line-length=88'
            }
        }
        
        stage('Build') {
            steps {
                sh 'docker build -t ml-project:${BUILD_NUMBER} .'
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'docker tag ml-project:${BUILD_NUMBER} your-registry/ml-project:latest'
                sh 'docker push your-registry/ml-project:latest'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'mlruns/**/*', fingerprint: true
        }
    }
}
