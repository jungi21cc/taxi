var size = {width: 710, height: 200};
// create our svg element
var svg = d3_svg.create('#example-1', size);

// build our chart
var chart = d3_line_chart.chart()
.width(size.width)
.height(size.height)
.margin({top: 10, left: 0, right: 0, bottom: 0})
.xValue(function(d) { return d.date; })
.yValue(function(d) { return d.val; });

// draw the chart
svg.datum(data).call(chart);

// add whatever we want
chart.g().append('text')
.text('much text')
.attr('dy', '1em');

chart.g().append('text')
.text('wow')
.attr('dy', '2em');
