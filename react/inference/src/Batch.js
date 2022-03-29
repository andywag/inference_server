import {Button, Grid, Box} from '@mui/material'
import { useState, useEffect } from 'react';


import BatchForm from './BatchForm'
import BatchTable from './BatchTable';

import ReactJson from 'react-json-view'

function Batch(props) {
    
    //const [data, setData] = useState([]);
    const [resultMap, setResultMap] = useState({});
    //const [selected, setSelected] = useState("");
    const [inferResults, setInferResults] = useState([]);
    const [description, setDescription] = useState({});

    /*
    var data = [];
    var description;
    if (resultMap.hasOwnProperty(selected)) {
        console.log("Selected", selected)
        console.log(resultMap[selected]);
        data = resultMap[selected].results;
        description = resultMap[selected].description;
    }
    */
    

    return (
        <Grid container spacing={2} >
            <Grid item xs={12}>
                <BatchForm/>
            </Grid>
            <Grid item xs={12}>
                <BatchTable setResultMap={setResultMap} setDescription={setDescription} setInferResults={setInferResults}/>
            </Grid>
            <Grid item xs={3}>
              <p><b>Configuration</b></p>
               <ReactJson collapsed="True" src={description}/>
            </Grid>

            <Grid item xs={3}>
                <p><b>Results</b></p>
                <ReactJson collapsed="True" src={inferResults}/>
            </Grid>
       </Grid>
    )

}

export default Batch;