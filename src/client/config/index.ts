
// Export default url config -- ideally would be dependentent on env variables
const config: any = {
    LOCAL: {
        apiUrl: 'http://localhost:9090'
    },
    DEV: {
        apiUrl: ''
    },
    PROD: {
        apiUrl: ''
    }
};

// Export the config
export default config;