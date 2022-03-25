import {Button, Grid} from '@mui/material'
import React, { useState, useEffect } from 'react';

import {Table,TableBody,TableCell,TableContainer,TableHead,TableRow} from '@mui/material';
import {Paper,Checkbox}  from '@mui/material';
import moment from 'moment';

import ReactJson from 'react-json-view'
import { DataGrid } from '@mui/x-data-grid';


function ResultTable(props) {
  

  const query = () => {
    const requestOptions = {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    };
    var time = new Date().getTime()

    fetch("http://192.168.3.114:8101/results",requestOptions)
      .then(response => response.json())
      .then(r => {
          var resultMap = {}
          r.forEach((element,index) => {
            resultMap[element.uuid] = element
          });
          
          setResults(r)
          props.setResultMap(resultMap)
          //props.setData(resultMap[selected].results)
        })
  }


  const [results, setResults] = useState([]);



  useEffect(() => {
    const timer = setInterval(query, 5000);
    return () => clearInterval(timer);
  }, []);




    const isSelected = (name) => {
        return props.selected == name
    };
    
    const handleClick = (evt, name) => {
        props.setSelected(name);
        //props.setData(resultMap[name].results)
    }

    const getDate = (row) => {
      const date = new Date(row.status[row.status.length-1].time * 1000)
      return moment(date).format('MM-DD-h:mm');
    }

    const getHost = (row) => {
      const value = row
      if (value.hasOwnProperty('hostname')) {
        return value.hostname
      }
      return ""
    }

    const handleContextMenu = () => {
      console.log("Here")
      
    }


    const columns = [
      { field: 'name', headerName: 'Name', width: 70 },
      { field: 'status', headerName: 'Status', width: 130 },
      { field: 'time', headerName: 'Time', width: 130 },
      { field: 'host', headerName: 'Host', width: 130 },
  
    ];


    const onSelection = (d,e) => {
      //console.log(results)
      //console.log(d)
      //console.log(results[d])
      //console.log(results[d].uuid)
      if (d.length > 0) {
        const selected = results[d].uuid;
        console.log("Selected", selected)
        props.setSelected(selected)
      }
      
    }

    const getRows = () => {
      const rr = results.map((row, index) => (
        { 
          id:index,
          name:row.description.name, 
          status:row.status[row.status.length-1].status,
          time: moment(new Date(row.status[row.status.length-1].time * 1000)).format('MM-DD-h:mm'),
          host:getHost(row)
      }));
      return rr;
      return [];
    };

    return (
      
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
    
    );

}

export default ResultTable;
