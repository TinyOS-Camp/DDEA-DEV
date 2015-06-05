function addAxesAndLegend (svg, xAxis, yAxis, margin, chartWidth, chartHeight) {
  var legendWidth  = 200,
      legendHeight = 100;

  // clipping to make sure nothing appears behind legend
  svg.append('clipPath')
    .attr('id', 'axes-clip')
    .append('polygon')
      .attr('points', (-margin.left)                 + ',' + (-margin.top)                 + ' ' +
                      (chartWidth - legendWidth - 1) + ',' + (-margin.top)                 + ' ' +
                      (chartWidth - legendWidth - 1) + ',' + legendHeight                  + ' ' +
                      (chartWidth + margin.right)    + ',' + legendHeight                  + ' ' +
                      (chartWidth + margin.right)    + ',' + (chartHeight + margin.bottom) + ' ' +
                      (-margin.left)                 + ',' + (chartHeight + margin.bottom));

  var axes = svg.append('g')
    .attr('clip-path', 'url(#axes-clip)');

  axes.append('g')
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,' + chartHeight + ')')
    .call(xAxis);

  axes.append('g')
    .attr('class', 'y axis')
    .call(yAxis)
    .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 6)
      .attr('dy', '.71em')
      .style('text-anchor', 'end')
      .text('Time (s)');

  var legend = svg.append('g')
    .attr('class', 'legend')
    .attr('transform', 'translate(' + (chartWidth - legendWidth) + ', 0)');

  legend.append('rect')
    .attr('class', 'legend-bg')
    .attr('width',  legendWidth)
    .attr('height', legendHeight);

  legend.append('rect')
    .attr('class', 'outer')
    .attr('width',  75)
    .attr('height', 20)
    .attr('x', 10)
    .attr('y', 10);

  legend.append('text')
    .attr('x', 115)
    .attr('y', 25)
    .text('5% - 95%');

  legend.append('rect')
    .attr('class', 'inner')
    .attr('width',  75)
    .attr('height', 20)
    .attr('x', 10)
    .attr('y', 40);

  legend.append('text')
    .attr('x', 115)
    .attr('y', 55)
    .text('25% - 75%');

  legend.append('path')
    .attr('class', 'median-line')
    .attr('d', 'M10,80L85,80');

  legend.append('text')
    .attr('x', 115)
    .attr('y', 85)
    .text('Median');
}

function drawPaths (svg, data, x, y) 
{

  var upperOuterArea = d3.svg.area()
    .interpolate('basis')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.pct95); })
    .y1(function (d) { return y(d.pct75); });

  var upperInnerArea = d3.svg.area()
    .interpolate('basis')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.pct75); })
    .y1(function (d) { return y(d.pct50); });

  var medianLine = d3.svg.line()
    .interpolate('basis')
    .x(function (d) { return x(d.date); })
    .y(function (d) { return y(d.pct50); });

  var lowerInnerArea = d3.svg.area()
    .interpolate('basis')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.pct50); })
    .y1(function (d) { return y(d.pct25); });

  var lowerOuterArea = d3.svg.area()
    .interpolate('basis')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.pct25); })
    .y1(function (d) { return y(d.pct05); });

  svg.datum(data);

  svg.append('path')
    .attr('class', 'area upper outer')
    .attr('d', upperOuterArea)
    .attr('clip-path', 'url(#rect-clip)');

  svg.append('path')
    .attr('class', 'area lower outer')
    .attr('d', lowerOuterArea)
    .attr('clip-path', 'url(#rect-clip)');

  svg.append('path')
    .attr('class', 'area upper inner')
    .attr('d', upperInnerArea)
    .attr('clip-path', 'url(#rect-clip)');

  svg.append('path')
    .attr('class', 'area lower inner')
    .attr('d', lowerInnerArea)
    .attr('clip-path', 'url(#rect-clip)');

  svg.append('path')
    .attr('class', 'median-line')
    .attr('d', medianLine)
    .attr('clip-path', 'url(#rect-clip)');
}

function addMarker (marker, svg, chartHeight, x) {
  var radius = 32,
      xPos = x(marker.date) - radius - 3,
      yPosStart = chartHeight - radius - 3,
      yPosEnd = (marker.type === 'Client' ? 80 : 160) + radius - 3;

  var markerG = svg.append('g')
    .attr('class', 'marker '+marker.type.toLowerCase())
    .attr('transform', 'translate(' + xPos + ', ' + yPosStart + ')')
    .attr('opacity', 0);

  markerG.transition()
    .duration(1000)
    .attr('transform', 'translate(' + xPos + ', ' + yPosEnd + ')')
    .attr('opacity', 1);

  markerG.append('path')
    .attr('d', 'M' + radius + ',' + (chartHeight-yPosStart) + 'L' + radius + ',' + (chartHeight-yPosStart))
    .transition()
      .duration(1000)
      .attr('d', 'M' + radius + ',' + (chartHeight-yPosEnd) + 'L' + radius + ',' + (radius*2));

  markerG.append('circle')
    .attr('class', 'marker-bg')
    .attr('cx', radius)
    .attr('cy', radius)
    .attr('r', radius);

  markerG.append('text')
    .attr('x', radius)
    .attr('y', radius*0.9)
    .text(marker.type);

  markerG.append('text')
    .attr('x', radius)
    .attr('y', radius*1.5)
    .text(marker.version);
}

function startTransitions (svg, chartWidth, chartHeight, rectClip, markers, x) {
  rectClip.transition()
    .duration(1000*markers.length)
    .attr('width', chartWidth);

  markers.forEach(function (marker, i) {
    setTimeout(function () {
      addMarker(marker, svg, chartHeight, x);
    }, 1000 + 500*i);
  });
}

function makeChart (data, markers) {
  var svgWidth  = 960,
      svgHeight = 500,
      margin = { top: 20, right: 20, bottom: 40, left: 40 },
      chartWidth  = svgWidth  - margin.left - margin.right,
      chartHeight = svgHeight - margin.top  - margin.bottom;

  var x = d3.time.scale().range([0, chartWidth]).domain(d3.extent(data, function (d) { return d.date; })),
      y = d3.scale.linear().range([chartHeight, 0]).domain([0, d3.max(data, function (d) { return d.pct95; })]);

  var xAxis = d3.svg.axis().scale(x).orient('bottom')
                .innerTickSize(-chartHeight).outerTickSize(0).tickPadding(10),
      yAxis = d3.svg.axis().scale(y).orient('left')
                .innerTickSize(-chartWidth).outerTickSize(0).tickPadding(10);

  var svg = d3.select('body').append('svg')
    .attr('width',  svgWidth)
    .attr('height', svgHeight)
    .append('g')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

  // clipping to start chart hidden and slide it in later
  var rectClip = svg.append('clipPath')
    .attr('id', 'rect-clip')
    .append('rect')
      .attr('width', 0)
      .attr('height', chartHeight);

  addAxesAndLegend(svg, xAxis, yAxis, margin, chartWidth, chartHeight);
  drawPaths(svg, data, x, y);
  startTransitions(svg, chartWidth, chartHeight, rectClip, markers, x);
}





//----------------------------------------------------------------------------------------------------------------------------------------------



function drawAxes (svg, xAxis, yAxis, margin, chartWidth, chartHeight) 
{
  var legendWidth  = 200,
      legendHeight = 100;

  var axes = svg.append('g')
    .attr('clip-path', 'url(#axes-clip)');

	axes.append('g')
		.attr('class', 'x axis')
		.attr('transform', 'translate('+ margin.left + ',' + chartHeight + ')')
		.call(xAxis);

  axes
    .append('g')
    .attr('class', 'y axis')
    .attr('transform', 'translate('+ margin.left + ', 0)')
    .call(yAxis)
    .append('text')
    .attr('transform', 'rotate(-90)')
    .attr('y', 6)
    .attr('dy', '.71em')
    .style('text-anchor', 'end')
    .text('Time (s)');
}

function plotFeature (svg, data, x, y)
{

  var diffLine = d3.svg.line().x(function (d) { return x(d.ts); }).y(function (d) { return y(d.diff); });
  var avgLine =  d3.svg.line().x(function (d) { return x(d.ts); }).y(function (d) { return y(d.avg); });

  svg.datum(data);

  svg.append('path')
    .attr('class', 'diff')
    .attr('d', diffLine)
    .attr('clip-path', 'url(#rect-clip)');

  svg.append('path')
    .attr('class', 'median-line')
    .attr('d', avgLine)
    .attr('clip-path', 'url(#rect-clip)');
}


function drawFeature (titleid, svgid, title, data) 
{
    //var svg = $("#feature-plot");
    //var svgWidth  = svg.prop("offsetWidth"),
    //    svgHeight = svg.prop("offsetHeight");

    $(titleid).text(title);

    var svg = $(svgid);
    while (svg.lastChild) {svg.removeChild(svg.lastChild);}
    svg.empty();

    svg = d3.select(svgid);

    var dimen = svg.node().getBoundingClientRect();

    var svgWidth  = dimen["width"],
        svgHeight = dimen["height"];

    console.log("svgHeight : " + svgWidth  + " svgHeight : " + svgHeight);

    var margin = { top: 20, right: 2, bottom: 30, left: 40 },
        chartWidth  = svgWidth  - margin.left - margin.right,
        chartHeight = svgHeight - margin.top  - margin.bottom;

    var x = d3.time.scale().range([0, chartWidth])
    		.domain(d3.extent(data, function (d) { return d.ts; })),

        y = d3.scale.linear().range([chartHeight, 0])
        	.domain([ d3.min(data, function (d) { return Math.min(d.avg, d.diff); }), 
        			  d3.max(data, function (d) { return Math.max(d.avg, d.diff); })  ]);

    var xAxis = d3.svg.axis().scale(x).orient('bottom').innerTickSize(-chartHeight).outerTickSize(0).tickPadding(10),
        yAxis = d3.svg.axis().scale(y).orient('left').innerTickSize(-chartWidth).outerTickSize(0).tickPadding(10);

    svg
    //.attr('width',  svgWidth).attr('height', svgHeight)
    .append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');


    // clipping to start chart hidden and slide it in later
    var rectClip = svg.append('clipPath')
                    .attr('id', 'rect-clip')
                    .append('rect')
                    .attr('x', margin.left)	
                    .attr('width', chartWidth)
                    .attr('height', chartHeight);

    drawAxes(svg, xAxis, yAxis, margin, chartWidth, chartHeight);
    plotFeature(svg, data, x, y);
}


//----------------------------------------------------------------------------------------------------------------------------------------------

function plotReading (svg, data, x, y)
{

  var readLine = d3.svg.line().x(function (d) { return x(d.ts); }).y(function (d) { return y(d.reading); });

  svg.datum(data);

  svg.append('path')
    .attr('class', 'median-line')
    .attr('d', readLine)
    .attr('clip-path', 'url(#rect-clip)');
}


function drawReading (titleid, svgid, title, data) 
{

    $(titleid).text(title);

    var svg = $(svgid);
    while (svg.lastChild) {svg.removeChild(svg.lastChild);}
    svg.empty();

    svg = d3.select(svgid);

    var dimen = svg.node().getBoundingClientRect();

    var svgWidth  = dimen["width"],
        svgHeight = dimen["height"];

    var margin = { top: 20, right: 2, bottom: 30, left: 60 },
        chartWidth  = svgWidth  - margin.left - margin.right,
        chartHeight = svgHeight - margin.top  - margin.bottom;

    var x = d3.time.scale().range([0, chartWidth])
        .domain(d3.extent(data, function (d) { return d.ts; })),

        y = d3.scale.linear().range([chartHeight, 0])
          .domain([ d3.min(data, function (d) { return d.reading; }), 
                    d3.max(data, function (d) { return d.reading; }) ]);

    var xAxis = d3.svg.axis().scale(x).orient('bottom').innerTickSize(-chartHeight).outerTickSize(0).tickPadding(10),
        yAxis = d3.svg.axis().scale(y).orient('left').innerTickSize(-chartWidth).outerTickSize(0).tickPadding(10);

    svg
    //.attr('width',  svgWidth).attr('height', svgHeight)
    .append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');


    // clipping to start chart hidden and slide it in later
    var rectClip = svg.append('clipPath')
                    .attr('id', 'rect-clip')
                    .append('rect')
                    .attr('x', margin.left) 
                    .attr('width', chartWidth)
                    .attr('height', chartHeight);

    drawAxes(svg, xAxis, yAxis, margin, chartWidth, chartHeight);
    plotReading(svg, data, x, y);
}


if(false)
{

  console.log("Start example")
  var parseDate  = d3.time.format('%Y-%m-%d').parse;
  d3.json('json/data.json', function (error, rawData) {
    if (error) {
      console.error(error);
      return;
    }

    var data = rawData.map(function (d) {
      return {
        date:  parseDate(d.date),
        pct05: d.pct05 / 1000,
        pct25: d.pct25 / 1000,
        pct50: d.pct50 / 1000,
        pct75: d.pct75 / 1000,
        pct95: d.pct95 / 1000
      };
    });

    d3.json('json/markers.json', function (error, markerData) {
      if (error) {
        console.error(error);
        return;
      }

      var markers = markerData.map(function (marker) {
        return {
          date: parseDate(marker.date),
          type: marker.type,
          version: marker.version
        };
      });

      makeChart(data, markers);
    });
  });
}

function heatmap(json_url, svg_id, colorCalibration)
{
    d3.json(json_url, function(err,data){

        //UI configuration
        var itemSize = 9,
            cellSize = itemSize-0.5;
            margin = {top:20,right:20,bottom:20,left:25};

        //data vars for rendering


        var svg = d3.select('[role="'+ svg_id + '"]');

        var sensor_state = data['sensor-state'];

        // all states have same length of elements
        var state_elem_length = sensor_state[0].length;
        var states_count = sensor_state.length;

        svg
            .style("height", itemSize * states_count);
/*
        svg
            .append("rect")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("fill", '#f00');
*/
        svg
            .selectAll("g")
            .data(sensor_state)
            .enter()
            .append("g")

            .selectAll("rect")
            .data(function(d,i,j) { return d; })
            .enter()
            .append("rect")

            .attr('width',cellSize)
            .attr('height',cellSize)
            .attr('x',function(d,i,j){
                return i * itemSize;
            })
            .attr('y',function(d,i,j){
                return j * itemSize;
            })
            .attr('fill',function(d,i,j){

                //choose color dynamicly
                var colorIndex = d3.scale
                    .quantize()
                    .range([0,1,2])
                    .domain([-1,1]);

                return colorCalibration[colorIndex(d)]
            });

        return;

        //render axises
        xAxis
            .scale(xAxisScale.range([0,axisWidth])
            .domain([dateExtent[0],dateExtent[1]]));

        svg.append('g')
            .attr('transform','translate(' + margin.left + ',' + margin.top + ')')
            .attr('class','x axis')
            .call(xAxis)
            .append('text')
            .text('date')
            .attr('transform','translate(' + axisWidth + ',-10)');

        svg.append('g')
            .attr('transform','translate(' + margin.left + ',' + margin.top + ')')
            .attr('class','y axis')
            .call(yAxis)
            .append('text')
            .text('time')
            .attr('transform','translate(-10,' + axisHeight + ') rotate(-90)');
    });
}