//Module imports
import * as bodyParser from "body-parser";
import * as cookieParser from "cookie-parser"
import * as express from "express";
import * as morgan from "morgan";
import * as path from "path";
import errorHandler = require("errorhandler");
import methodOverride = require("method-override");

import { IndexRouter } from "./routers";

//The Server class
export class Server
{
    public app : express.Application;

    //Constuctor for the server
    public constructor()
    {
        //Create an ExpressJS application instance
        this.app = express();

        //Configure the application
        this.Configure();

        //Add the routes
        this.Routes();

        //Add the API
        this.API();
    }

    //Create REST API routes?
    public API()
    {

    }

    //Configure the Application
    public Configure()
    {
        //Add static paths -- needs to be updated for the different frontend methods
        //this.app.use(express.static(path.join(__dirname, "./views/Vue")));
        this.app.use(express.static(path.join(__dirname, "../../../../", "dist")));

        //Use Logger middleware
        this.app.use(morgan("dev"));

        //Use JSON form parsing
        this.app.use(bodyParser.json());
        //Use Query string parsing
        this.app.use(bodyParser.urlencoded({ extended : true }));

        //Use Cookie Parser
        this.app.use(cookieParser("SELECT_GOES_HERE"));

        //Use Override
        this.app.use(methodOverride());

        //Catch 404 error and forward to error handler
        this.app.use(function(error : any, request : express.Request, response : express.Response, next : express.NextFunction)
        {
            error.status = 404;
            next(error);
        });

        //Add error handler
        this.app.use(errorHandler());
    }

    private Routes()
    {
        //Index router for homepage and related
        const indexRouter: IndexRouter = new IndexRouter("This is the homepage router");

        //Use the router middleware -- use as many as necessary -- assign them their own base addresses for relative paths
        this.app.use('/', indexRouter.router);
    }
}