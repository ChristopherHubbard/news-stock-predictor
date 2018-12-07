
// Export default url config -- ideally would be dependentent on env variables
const config: any = {
    LOCAL: {
        apiUrl: 'http://localhost:9090'
    },
    DEV: {
        apiUrl: 'https://n5kiwn5jok.execute-api.us-east-2.amazonaws.com/dev'
    },
    PROD: {
        apiUrl: ''
    }
};

// Export the config
export default config;