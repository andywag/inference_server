import {Button, Grid} from '@mui/material'
import React, { useState, useEffect } from 'react';

import {Table,TableBody,TableCell,TableContainer,TableHead,TableRow} from '@mui/material';
import {Paper,Checkbox}  from '@mui/material';
import moment from 'moment';

import ReactJson from 'react-json-view'
import { DataGrid } from '@mui/x-data-grid';


function FineTable(props) {
  
  const [results, setResults] = useState([]);


  const requestOptions = {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  };
  const query = () => {
    
    var time = new Date().getTime()

    fetch("http://192.168.3.114:8101/results",requestOptions)
      .then(response => response.json())
      .then(r => {
          var resultMap = {}
          r.forEach((element,index) => {
            resultMap[element.uuid] = element
          });
          
          setResults(r)
          //props.setResultMap(resultMap)
        })
  }

  /*
  const queryResults = (selected) => {
    
    console.log("Query Results", selected);
    const qu = 'http://192.168.3.114:8101/infer_final_result?id=' + selected
    fetch(qu,requestOptions)
      .then(response => response.json())
      .then(r => {
        //console.log(r)
        if (r) {
          props.setInferResults(r);
        }
        else {
          props.setInferResults({});
        }
          
        })
  }
  */


  useEffect(() => {
    const timer = setInterval(query, 5000);
    return () => clearInterval(timer);
  }, []);



    const getHost = (row) => {
      const value = row
      if (value.hasOwnProperty('hostname')) {
        return value.hostname
      }
      return ""
    }



    const columns = [
      { field: 'name', headerName: 'Name', width: 100 },
      { field: 'status', headerName: 'Status', width: 130 },
      { field: 'time', headerName: 'Time', width: 220 },
      { field: 'host', headerName: 'Host', width: 130 },
      { field: 'loss', headerName: 'Loss', width: 130 },
      { field: 'qps', headerName: 'QPS', width: 130 }
    ];


    const onSelection = (d,e) => {
      if (d.length > 0) {
        console.log(results[d[0]]);
        //const selected = results[d[0]].sid;
        props.setDescription(results[d[0]].description);
        props.setResults(results[d[0]].results);

      }
      
    }

    const getName = (row) => {
      if (row.description) {
        return row.description.name
      }
      return ""
  }


    const getTime = (row) => {
      if (row.status && row.status.length > 0) {
        return new Date(row.status[0].time * 1000)
      }
      return ""
  }

    const getStatus = (row) => {
        if (row.status && row.status.length > 0) {
          return row.status[row.status.length-1].status
        }
        return ""
    }



    const getRows = () => {
      const rr = results.map((row, index) => (
        { 
          id:index,
          name:getName(row), 
          status:getStatus(row),
          time: getTime(row),
          host:getHost(row),
          loss:row.loss,
          qps:row.qps
      }));
      return rr;
      return [];
    };

    return (
      <div style={{ height: 350, width: '100%' }}>
        <DataGrid
        rows={getRows()}
        columns={columns}
        pageSize={5}
        rowsPerPageOptions={[5]}
        checkboxSelection
        onSelectionModelChange={onSelection}
        initialState={{
          sorting: {
            sortModel: [{ field: 'time'}],
          },
        }}
      />
      </div>
      
    
    );

}

export default FineTable;
