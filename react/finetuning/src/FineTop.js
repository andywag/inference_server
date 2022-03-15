import {Button, Grid, Box} from '@mui/material'
import { useState, useEffect } from 'react';


import FineForm from './FineForm'
import ResultTable from './ResultTable'
import ResultChart from './ResultChart'

function FineTop(props) {
    
    //const [data, setData] = useState([]);
    const [resultMap, setResultMap] = useState({});
    const [selected, setSelected] = useState("");

    var data = [];
    if (resultMap.hasOwnProperty(selected)) {
        data = resultMap[selected].results;
    }
    

    return (
        <Grid container spacing={2} >
            <Grid item xs={12}>
                <FineForm/>
            </Grid>
            <Grid item xs={6}>
                <ResultTable setResultMap= {setResultMap} selected={selected} setSelected={setSelected}/>
            </Grid>
            <Grid item xs={6}>
                <ResultChart data = {data}/>
            </Grid>
       </Grid>
    )

}

export default FineTop;