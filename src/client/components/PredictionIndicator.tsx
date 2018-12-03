import * as React from 'react';
import { connect, DispatchProp } from 'react-redux';
import { Prediction } from '../models/prediction.model';

// Have to import images for webpack
const incIcon = require('../assets/predictionIncrease.png');
const decIcon = require('../assets/predictionDecrease.png');

interface PredictionIndicatorProps
{
    prediction: Prediction,
    predicting: boolean
}

class PredictionIndicator extends React.Component<PredictionIndicatorProps & DispatchProp<any>, {}>
{
    constructor(props: PredictionIndicatorProps & DispatchProp<any>)
    {
        super(props);
    }

    public render(): React.ReactNode
    {
        // Extract prop data -- handle the case when still predicting
        const { prediction, predicting } = this.props;

        let graphic: JSX.Element | undefined = undefined;
        if (prediction === Prediction.Up)
        {
            graphic = <img src={incIcon}/> 
        }
        else if (prediction === Prediction.Down)
        {
            graphic = <img src={decIcon}/>
        }

        // Render the indicator for if the index is increasing -- also indicator for loading
        return (
            <div>
                {graphic}
            </div>
        )
    }
}

function mapStateToProps(state: any): PredictionIndicatorProps
{
    // Extract the state from the action
    const { prediction, predicting } = state.prediction;
    return {
        prediction,
        predicting
    };
}

export default connect<PredictionIndicatorProps>(
    mapStateToProps
)(PredictionIndicator);