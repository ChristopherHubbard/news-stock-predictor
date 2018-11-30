//Module imports
import { NextFunction, Request, Response, Router } from "express";

//Route Class
export abstract class CustomRouter
{
    public router: Router;

    protected title : string;

    private scripts : string[];

    //Constructor
    public constructor(title : string)
    {
        this.title = title;
        this.scripts = [];
        this.router = Router();

        // Create the routes? Should be defined in the sub class, so this should call that method?
        this.CreateRoutes();
    }

    //Add a TS external file to the request -- given the source to the external file and returns this object for chaining
    public AddScript(src : string) : CustomRouter
    {
        this.scripts.push(src);
        return this;
    }

    //Render a page -- is this necessary with React??
    public Render(request : Request, response : Response, view : string, options? : Object)
    {
        response.locals.Base_URL = "/";

        //Add scripts
        response.locals.scripts = this.scripts;

        //Add title
        response.locals.title = this.title;

        //Render the view -- ??
        response.render(view, options);
    }

    // Method required to be implemented by the class to define the routes that the router uses
    protected abstract CreateRoutes(): void
}