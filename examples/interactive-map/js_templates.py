"""
JavaScript templates for interactive visualizations.
"""

# JavaScript template for click-to-highlight functionality
HIGHLIGHT_SCRIPT_TEMPLATE = """
    <script>
        // Pairings data: maps "traceIdx_pointIdx" to array of {{traceIndex, pointIndex}} objects
        const pairingsData = {pairings_json};
        
        // Highlight style configuration
        const highlightStyle = {highlight_style_json};
        
        // State tracking - stores only modified points for properties we change
        let highlightState = {{
            activeSourceKey: null,
            modifiedPoints: {{}}  // {{traceIdx: {{pointIdx: {{width, color, size?}}}}}}
        }};
        
        // Helper function to check if a property is already an array (per-point)
        function isArrayProperty(property) {{
            return property != null && typeof property === 'object' && property.length !== undefined;
        }}
        
        // Helper function to clone marker property (scalar or array)
        function cloneMarkerProperty(property, numPoints) {{
            if (property == null) {{
                return new Array(numPoints).fill(0);
            }}
            if (typeof property === 'object' && property.length !== undefined) {{
                return Array.from(property);
            }}
            // Scalar - create filled array
            return new Array(numPoints).fill(property);
        }}
        
        // Wait for DOM and Plotly plot to be ready
        window.addEventListener('load', function() {{
            const plotDiv = document.getElementById('protein-domain-map');
            
            if (!plotDiv) {{
                console.error('Plot container not found');
                return;
            }}
            
            // Attach click handler
            plotDiv.on('plotly_click', function(data) {{
                const clickedPoint = data.points[0];
                const sourceKey = clickedPoint.curveNumber + '_' + clickedPoint.pointIndex;
                
                // Check if this source has pairings
                if (!pairingsData[sourceKey]) {{
                    return;
                }}
                
                // Toggle behavior
                if (highlightState.activeSourceKey === sourceKey) {{
                    resetHighlight();
                }} else {{
                    if (highlightState.activeSourceKey) {{
                        resetHighlight();
                    }}
                    highlightTargets(sourceKey, pairingsData[sourceKey]);
                }}
            }});
        
            function highlightTargets(sourceKey, targets) {{
                highlightState.activeSourceKey = sourceKey;
                highlightState.modifiedPoints = {{}};
                
                // Group targets by trace for efficient batched updates
                const traceUpdates = {{}};
                
                targets.forEach(target => {{
                    const traceIdx = target.traceIndex;
                    const pointIdx = target.pointIndex;
                    
                    if (!traceUpdates[traceIdx]) {{
                        const trace = plotDiv.data[traceIdx];
                        const numPoints = trace.x.length;
                        
                        // Always modify marker.line.* properties
                        traceUpdates[traceIdx] = {{
                            lineWidths: cloneMarkerProperty(trace.marker.line.width, numPoints),
                            lineColors: cloneMarkerProperty(trace.marker.line.color, numPoints),
                            shouldModifySize: isArrayProperty(trace.marker.size)
                        }};
                        
                        // ONLY modify marker.size if it's already an array (per-point)
                        if (traceUpdates[traceIdx].shouldModifySize) {{
                            traceUpdates[traceIdx].sizes = cloneMarkerProperty(trace.marker.size, numPoints);
                        }}
                        
                        // Initialize storage for this trace's modified points
                        highlightState.modifiedPoints[traceIdx] = {{}};
                    }}
                    
                    // Store original values for this specific point
                    const originalValues = {{
                        width: traceUpdates[traceIdx].lineWidths[pointIdx],
                        color: traceUpdates[traceIdx].lineColors[pointIdx]
                    }};
                    
                    if (traceUpdates[traceIdx].shouldModifySize) {{
                        originalValues.size = traceUpdates[traceIdx].sizes[pointIdx];
                    }}
                    
                    highlightState.modifiedPoints[traceIdx][pointIdx] = originalValues;
                    
                    // Apply highlight styling
                    traceUpdates[traceIdx].lineWidths[pointIdx] = highlightStyle.lineWidth;
                    traceUpdates[traceIdx].lineColors[pointIdx] = highlightStyle.lineColor;
                    
                    // Only modify size if it's safe to do so
                    if (traceUpdates[traceIdx].shouldModifySize) {{
                        traceUpdates[traceIdx].sizes[pointIdx] = highlightStyle.size;
                    }}
                }});
                
                // Apply all updates in a single batched operation
                const updateData = {{}};
                const traceIndices = [];
                
                Object.keys(traceUpdates).forEach(traceIdx => {{
                    const update = traceUpdates[traceIdx];
                    const idx = traceIndices.length;
                    traceIndices.push(parseInt(traceIdx));
                    
                    if (!updateData['marker.line.width']) {{
                        updateData['marker.line.width'] = [];
                        updateData['marker.line.color'] = [];
                    }}
                    
                    updateData['marker.line.width'][idx] = update.lineWidths;
                    updateData['marker.line.color'][idx] = update.lineColors;
                    
                    // Only update marker.size if we determined it's safe
                    if (update.shouldModifySize) {{
                        if (!updateData['marker.size']) {{
                            updateData['marker.size'] = [];
                        }}
                        updateData['marker.size'][idx] = update.sizes;
                    }}
                }});
                
                // Single Plotly.restyle call for all traces
                if (traceIndices.length > 0) {{
                    Plotly.restyle(plotDiv, updateData, traceIndices);
                }}
            }}
            
            function resetHighlight() {{
                if (!highlightState.activeSourceKey) {{
                    return;
                }}
                
                // Restore only the modified points
                const updateData = {{}};
                const traceIndices = [];
                
                Object.keys(highlightState.modifiedPoints).forEach(traceIdx => {{
                    const trace = plotDiv.data[traceIdx];
                    const numPoints = trace.x.length;
                    const pointsToRestore = highlightState.modifiedPoints[traceIdx];
                    
                    // Clone current arrays - ONLY the properties we modified
                    const lineWidths = cloneMarkerProperty(trace.marker.line.width, numPoints);
                    const lineColors = cloneMarkerProperty(trace.marker.line.color, numPoints);
                    
                    const shouldRestoreSize = isArrayProperty(trace.marker.size);
                    let sizes;
                    if (shouldRestoreSize) {{
                        sizes = cloneMarkerProperty(trace.marker.size, numPoints);
                    }}
                    
                    // Restore only the modified points
                    Object.keys(pointsToRestore).forEach(pointIdx => {{
                        const originalValues = pointsToRestore[pointIdx];
                        lineWidths[pointIdx] = originalValues.width;
                        lineColors[pointIdx] = originalValues.color;
                        
                        if (shouldRestoreSize && originalValues.size !== undefined) {{
                            sizes[pointIdx] = originalValues.size;
                        }}
                    }});
                    
                    const idx = traceIndices.length;
                    traceIndices.push(parseInt(traceIdx));
                    
                    if (!updateData['marker.line.width']) {{
                        updateData['marker.line.width'] = [];
                        updateData['marker.line.color'] = [];
                    }}
                    
                    updateData['marker.line.width'][idx] = lineWidths;
                    updateData['marker.line.color'][idx] = lineColors;
                    
                    if (shouldRestoreSize) {{
                        if (!updateData['marker.size']) {{
                            updateData['marker.size'] = [];
                        }}
                        updateData['marker.size'][idx] = sizes;
                    }}
                }});
                
                // Single Plotly.restyle call for all traces
                if (traceIndices.length > 0) {{
                    Plotly.restyle(plotDiv, updateData, traceIndices);
                }}
                
                // Clear state
                highlightState.activeSourceKey = null;
                highlightState.modifiedPoints = {{}};
            }}
        }});
    </script>
"""
