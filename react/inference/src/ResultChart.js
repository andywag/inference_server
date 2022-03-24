import {Grid} from '@mui/material'
import { LineChart, XAxis, YAxis, CartesianGrid, Line} from 'recharts';
import ReactJson from 'react-json-view'

function ResultChart(props) {
  
    return (
        <Grid item xs={6}>
            
            <LineChart width={500} height={300} data={props.data}>
                <XAxis/>
                <YAxis/>
                <CartesianGrid stroke="#eee" strokeDasharray="5 5"/>
                <Line type="monotone" dataKey="error" stroke="#8884d8" />
            </LineChart>
            <ReactJson collapsed="True" src={props.description}/>
        </Grid>
    );

}

export default ResultChart;
