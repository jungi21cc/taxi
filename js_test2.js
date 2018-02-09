requirejs.config({
    paths: {
        vis: 'vis'
    }
});

require(['vis'], function(vis){

    var nodes = [
        {id: 1, label: 'Beyonce', group: 'United States'},
        {id: 2, label: 'Barak Obama', group: 'United States'},
        {id: 3, label: 'Miley Cyrus', group: 'United States'},
        {id: 4, label: 'Pope Francis', group: 'Vatican'},
        {id: 5, label: 'Vladimir Putin', group: 'Rusia'}
    ];

    // create an array with edges
    var edges = [
        {from: 1, to: 2},
        {from: 1, to: 3},
        {from: 2, to: 4},
        {from: 2, to: 5}
    ];

    // create a network
    var container = document.getElementById('mynetwork');
    var data= {
        nodes: nodes,
        edges: edges,
    };
    var options = {
        width: '800px',
        height: '400px'
    };

    var network = new vis.Network(container, data, options);
});
