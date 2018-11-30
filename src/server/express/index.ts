//Module Dependencies
import { Server } from "./Server";
let debug : any = require("debug")("express:server");
let http : any = require("http");

//Set the app
let app : any = (new Server()).app;
//Get the port from the enviornment
const port : string | number | boolean = NormalizePort(process.env.PORT || 8080);
app.set("port", port);

//Create http server
let httpServer : any = http.createServer(app);

//Listen on the provided ports
httpServer.listen(port);

//Add an error handler
httpServer.on("error", OnError);

httpServer.on("listening", OnListening);

//Function to normalize a port into a number, string, or false
function NormalizePort(value : number | string) : number | string | boolean
{
    let port : number = (typeof value === "string") ? parseInt(value, 10) : value;

    if (isNaN(port))
    {   
        return value
    }
    else if (port >= 0)
    {
        return port;
    }
    else
    {
        return false;
    }
}

//Function when an error occurs
function OnError(error : NodeJS.ErrnoException): void 
{
    if (error.syscall !== "listen") 
    {
        throw error;
    }

    let bind : string = (typeof port === "string") ? "Pipe " + port : "Port " + port;

    switch (error.code) 
    {
        case 'EACCES':
            console.error(`${bind} requires elevated privileges`);
            process.exit(1);
            break;
        case 'EADDRINUSE':
            console.error(`${bind} is already in use`);
            process.exit(1);
            break;
        default:
            throw error;
    }
}

//Function to listen to the http listening event
function OnListening() : void
{
    let address : any = httpServer.address();
    let bind : string = (typeof address === "string") ? "pipe " + address : "port " + address.port;
    debug("Listening on " + bind);
}