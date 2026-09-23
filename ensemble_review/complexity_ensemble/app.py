"""Gradio front end for regional CAMELS-CH experiments."""

from __future__ import annotations

import argparse
import html
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .regional import (
    BasinScopeConfig, CAMELSCHCatalog, CancellationToken, DateRange, DateSplitConfig,
    ExtremeEventConfig, HyperparameterConfig, ModelArchitectureConfig,
    PhysicsModelConfig, RegionalExperimentConfig, TrainingStrategyConfig,
    run_regional_experiment, write_result_artifacts,
)


DEFAULT_BASINS = (
    "4009", "4008", "2488", "2109", "2011", "2126", "5001", "2247",
    "4022", "4023", "5009", "5010", "5016", "3014", "3015",
)


_LEAFLET_HEAD = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
"""

_BASIN_MAP_TEMPLATE = """
<div class="basin-map-shell">
  <div class="basin-map-canvas" role="application" aria-label="Interactive CAMELS-CH basin map"></div>
  <div class="basin-map-legend" aria-label="Map legend">
    <span><i class="target"></i>Prediction target</span>
    <span><i class="training"></i>Training basin</span>
    <span><i class="held-out"></i>Held-out target</span>
    <span><i class="unused"></i>Unused / ineligible</span>
  </div>
</div>
"""

_BASIN_MAP_CSS = """
.basin-map-shell { position: relative; width: 100%; }
.basin-map-canvas {
  width: 100%; height: 520px; border: 1px solid var(--border-color-primary);
  border-radius: 10px; overflow: hidden; background: #edf3f7;
}
.basin-map-legend {
  position: absolute; z-index: 500; left: 12px; bottom: 12px;
  display: flex; flex-wrap: wrap; gap: 6px 12px; max-width: calc(100% - 24px);
  padding: 7px 10px; border-radius: 7px; background: rgba(255,255,255,.92);
  box-shadow: 0 1px 5px rgba(15,23,42,.22); color: #1e293b; font-size: 12px;
}
.basin-map-legend span { display: inline-flex; align-items: center; gap: 5px; }
.basin-map-legend i { width: 11px; height: 11px; border: 1px solid #475569; border-radius: 2px; }
.basin-map-legend .target { background: #2563eb; }
.basin-map-legend .training { background: #16a34a; }
.basin-map-legend .held-out { background: #f59e0b; }
.basin-map-legend .unused { background: #cbd5e1; }
.basin-map-shell .leaflet-container { font-family: inherit; }
.basin-map-shell .leaflet-tooltip { color: #0f172a; }
.basin-map-error { padding: 2rem; color: #991b1b; }
"""

_BASIN_MAP_JS = r"""
const canvas = element.querySelector('.basin-map-canvas');
if (!window.L) {
    canvas.innerHTML = '<div class="basin-map-error">The interactive map library could not be loaded. Check the browser network connection and reload the page.</div>';
    return;
}

const map = L.map(canvas, {
    zoomControl: true,
    attributionControl: false,
    preferCanvas: true,
    minZoom: 6,
    maxZoom: 14,
});
map.createPane('basins');
map.getPane('basins').style.zIndex = 410;
let basinLayer = null;

const escapeHtml = (raw) => String(raw ?? '').replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[char]));

const state = () => props.value || {};
const styleFeature = (feature) => {
    const current = state();
    const id = String(feature.properties.basin_id);
    const selected = new Set((current.selected || []).map(String));
    const training = new Set((current.training || []).map(String));
    let fillColor = '#cbd5e1';
    if (current.future) fillColor = selected.has(id) ? '#2563eb' : '#cbd5e1';
    else if (training.has(id)) fillColor = '#16a34a';
    else if (selected.has(id)) fillColor = '#f59e0b';
    return {
        pane: 'basins', color: id === String(current.active ?? '') ? '#0f172a' : '#475569',
        weight: id === String(current.active ?? '') ? 4 : 1.2,
        fillColor, fillOpacity: selected.has(id) || training.has(id) ? 0.72 : 0.42,
    };
};

const selectBasin = (id) => {
    const current = state();
    if (current.interaction === 'none') return;
    if (current.interaction === 'activate') {
        if (!(current.selected || []).map(String).includes(id)) return;
        props.value = {...current, active: id};
    } else {
        if (!(current.eligible || []).map(String).includes(id)) return;
        const selected = (current.selected || []).map(String);
        props.value = {...current, selected: selected.includes(id)
            ? selected.filter((item) => item !== id) : [...selected, id]};
    }
    if (basinLayer) basinLayer.setStyle(styleFeature);
    trigger('input');
};

const makeLayer = () => {
    const current = state();
    basinLayer = L.geoJSON(current.geojson || {type: 'FeatureCollection', features: []}, {
        style: styleFeature,
        onEachFeature: (feature, layer) => {
            const details = feature.properties || {};
            layer.bindTooltip(
                `<strong>${escapeHtml(details.basin_id)}</strong>` +
                `${details.name ? '<br>' + escapeHtml(details.name) : ''}` +
                `<br>${details.eligible ? 'Eligible' : 'Ineligible'}`,
                {sticky: true}
            );
            layer.on('click', () => selectBasin(String(details.basin_id)));
            layer.on('mouseover', () => layer.setStyle({weight: 3, fillOpacity: .82}));
            layer.on('mouseout', () => basinLayer.resetStyle(layer));
        },
    }).addTo(map);
    if (basinLayer.getLayers().length) map.fitBounds(basinLayer.getBounds(), {padding: [18, 18]});
    else {
        map.setView([46.8, 8.2], 7);
        canvas.insertAdjacentHTML('beforeend', '<div class="basin-map-error">No basin polygons were found in the CAMELS-CH data root.</div>');
    }
};

makeLayer();
L.control.scale({imperial: false, position: 'bottomright'}).addTo(map);
setTimeout(() => map.invalidateSize(), 0);
watch('value', () => {
    if (basinLayer) basinLayer.setStyle(styleFeature);
});
"""


@dataclass
class BasinMapComponent:
    """Reusable, tile-free SVG renderer and selection-state helper."""

    catalog: CAMELSCHCatalog

    def toggle(self, selected: Iterable[str], basin_id: str) -> list[str]:
        result = list(dict.fromkeys(map(str, selected)))
        if basin_id in result:
            result.remove(basin_id)
        elif basin_id in {item.basin_id for item in self.catalog.records if item.eligible}:
            result.append(basin_id)
        return result

    def payload(self, selected: Iterable[str] = (), active: str | None = None,
                training: Iterable[str] = (), future: bool = False,
                interaction: str = "none") -> dict[str, object]:
        """Return the serializable state consumed by the browser map."""
        features = []
        records = sorted(
            (record for record in self.catalog.records if record.geometry is not None),
            key=lambda record: getattr(record.geometry, "area", 0.0), reverse=True,
        )
        for record in records:
            interface = getattr(record.geometry, "__geo_interface__", None)
            if interface is None:
                continue
            features.append({
                "type": "Feature",
                "id": record.basin_id,
                "properties": {
                    "basin_id": record.basin_id,
                    "name": record.name,
                    "eligible": record.eligible,
                },
                "geometry": interface,
            })
        return {
            "geojson": {"type": "FeatureCollection", "features": features},
            "eligible": list(self.catalog.eligible_ids),
            "selected": list(dict.fromkeys(map(str, selected))),
            "training": list(dict.fromkeys(map(str, training))),
            "active": str(active) if active is not None else None,
            "future": future,
            "interaction": interaction,
        }

    def gradio_component(self, gr, *, selected: Iterable[str] = (),
                         active: str | None = None, training: Iterable[str] = (),
                         future: bool = False, interaction: str = "none",
                         label: str | None = None):
        """Create a Gradio 6 HTML component backed by a Leaflet vector map."""
        return gr.HTML(
            value=self.payload(selected, active, training, future, interaction),
            label=label,
            show_label=label is not None,
            html_template=_BASIN_MAP_TEMPLATE,
            css_template=_BASIN_MAP_CSS,
            js_on_load=_BASIN_MAP_JS,
            head=_LEAFLET_HEAD,
            apply_default_css=False,
            container=True,
            padding=False,
        )

    def svg(self, selected: Iterable[str] = (), active: str | None = None,
            training: Iterable[str] = (), future: bool = False) -> str:
        selected, training = set(map(str, selected)), set(map(str, training))
        geometries = [(record, record.geometry) for record in self.catalog.records
                      if record.geometry is not None and hasattr(record.geometry, "bounds")]
        if not geometries:
            labels = "".join(
                f'<text x="10" y="{22 + index*18}" data-basin="{html.escape(record.basin_id)}">'
                f'{html.escape(record.basin_id)} — {html.escape(record.name)}</text>'
                for index, record in enumerate(self.catalog.records[:25])
            )
            return f'<svg viewBox="0 0 500 480" role="img" aria-label="CAMELS-CH basin map">{labels}</svg>'
        minx = min(shape.bounds[0] for _, shape in geometries); miny = min(shape.bounds[1] for _, shape in geometries)
        maxx = max(shape.bounds[2] for _, shape in geometries); maxy = max(shape.bounds[3] for _, shape in geometries)
        scale = min(760/max(maxx-minx, 1e-9), 480/max(maxy-miny, 1e-9))
        paths = []
        for record, shape in geometries:
            polygons = getattr(shape, "geoms", (shape,))
            commands = []
            for polygon in polygons:
                exterior = getattr(polygon, "exterior", None)
                if exterior is None: continue
                points = [((x-minx)*scale+10, 500-(y-miny)*scale) for x, y in exterior.coords]
                commands.append("M " + " L ".join(f"{x:.2f},{y:.2f}" for x,y in points) + " Z")
            if future:
                color = "#2563eb" if record.basin_id in selected else "#d1d5db"
            else:
                color = "#16a34a" if record.basin_id in training else ("#f59e0b" if record.basin_id in selected else "#d1d5db")
            width = 3 if record.basin_id == active else 1
            paths.append(f'<path d="{" ".join(commands)}" fill="{color}" stroke="#334155" stroke-width="{width}" data-basin="{html.escape(record.basin_id)}"><title>{html.escape(record.basin_id)} — {html.escape(record.name)}</title></path>')
        return '<svg viewBox="0 0 800 520" role="img" aria-label="CAMELS-CH basin map" style="width:100%;max-height:520px">' + "".join(paths) + '</svg>'


def _parse_seeds(value: str) -> tuple[int, ...]:
    try: return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error: raise ValueError("seeds must be comma-separated integers") from error


def _config_from_ui_values(data_root: str | Path, target_values, scope_value, values):
    """Translate the ordered Gradio component values into the typed service config."""
    widths = tuple(int(item.strip()) for item in values[22].split(",") if item.strip())
    return RegionalExperimentConfig(
        data_root,
        BasinScopeConfig(tuple(target_values or ()), scope_value),
        DateSplitConfig(DateRange(values[0], values[1]), DateRange(values[2], values[3]),
                        DateRange(values[4], values[5])),
        ModelArchitectureConfig(
            complex_expert=values[7], simple_expert=values[8], mlp_widths=widths,
            mamba_hidden_size=int(values[23]), mamba_layers=int(values[24]),
            mamba_feed_forward_size=int(values[25]), rbf_centers=int(values[26]),
            fourier_frequencies=int(values[27]),
        ),
        TrainingStrategyConfig(values[6], int(values[9]), int(values[19]),
                               int(values[20]), int(values[21])),
        HyperparameterConfig(
            int(values[10]), _parse_seeds(values[11]), int(values[12]), float(values[13]),
            float(values[14]), float(values[15]), float(values[16]), float(values[17]),
            float(values[18]), float(values[29]), float(values[30]), values[31],
        ),
        PhysicsModelConfig(),
        ExtremeEventConfig(values[28], values[32],
                           float(values[33]) if values[33] is not None else None),
    )


def build_app(data_root: str | Path):
    """Build a session-scoped Gradio Blocks application."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is required to build the application") from error
    catalog = CAMELSCHCatalog(data_root)
    map_component = BasinMapComponent(catalog)
    choices = [(f"{item.basin_id} — {item.name}", item.basin_id) for item in catalog.records if item.eligible]
    eligible = set(catalog.eligible_ids)
    default_basins = [basin for basin in DEFAULT_BASINS if basin in eligible]

    with gr.Blocks(title="Regional Hydrology") as app:
        token_state = gr.State(None)
        experiment_state = gr.State(None)
        gr.Markdown("# Regional hydrology experiment")
        with gr.Tab("Target basins"):
            basin_map = map_component.gradio_component(
                gr, selected=default_basins, future=True, interaction="toggle",
                label="Select prediction targets on the map",
            )
            targets = gr.Dropdown(
                choices=choices, value=default_basins, multiselect=True,
                label="Prediction targets",
            )
            with gr.Row():
                select_all = gr.Button("Select all eligible"); clear = gr.Button("Clear")
            scope = gr.Radio([
                ("All eligible except targets", "exclude_targets"),
                ("All eligible (targets included)", "all_basins"),
                ("Only selected targets", "targets_only"),
            ], value="all_basins", label="Model-training basins")
            eligibility = gr.Dataframe(headers=["basin_id", "name", "eligible"],
                value=[[x.basin_id, x.name, x.eligible] for x in catalog.records], interactive=False)
        with gr.Tab("Experiment configuration"):
            with gr.Row():
                train_start=gr.Textbox("2010-01-01",label="Train start"); train_end=gr.Textbox("2016-08-06",label="Train end")
                val_start=gr.Textbox("2016-08-07",label="Validation start"); val_end=gr.Textbox("2018-10-18",label="Validation end")
                test_start=gr.Textbox("2018-10-19",label="Test start"); test_end=gr.Textbox("2020-12-31",label="Test end")
            approach=gr.Radio([("Soft routing","soft_routing"),("Hard routing","hard_routing"),("Distillation","distillation")],value="soft_routing",label="Approach")
            complex_expert=gr.Radio([("Mamba","pinnmamba"),("MLP","mlp")],value="pinnmamba",label="Complex expert")
            simple_expert=gr.Radio([("RBF","rbf"),("Fourier","fourier")],value="rbf",label="Simple expert")
            gr.Dropdown([("Linear reservoir water balance","linear_reservoir")],value="linear_reservoir",label="Physics model",interactive=False)
        with gr.Tab("Training hyperparameters"):
            with gr.Row():
                sequence=gr.Number(30,precision=0,label="Sequence length"); seeds=gr.Textbox("0",label="Seeds")
                epochs=gr.Number(20,precision=0,label="Epochs"); batch=gr.Number(64,precision=0,label="Batch size")
                learning_rate=gr.Number(1e-3,label="Learning rate"); noise=gr.Number(0.0,label="Training noise")
            with gr.Row():
                percentile=gr.Number(80,label="Complexity percentile"); temperature=gr.Number(.15,label="Gate temperature")
                physics_weight=gr.Number(.05,label="Physics weight"); interface_weight=gr.Number(.05,label="Interface weight")
                routing_weight=gr.Number(.05,label="Routing weight"); compute_weight=gr.Number(0,label="Compute weight")
            device=gr.Radio(["cpu","cuda"],value="cpu",label="Device")
            with gr.Accordion("Approach and architecture controls",open=False):
                teacher_epochs=gr.Number(20,precision=0,label="Complex-teacher epochs")
                distill_epochs=gr.Number(20,precision=0,label="Distillation epochs")
                consolidation_epochs=gr.Number(20,precision=0,label="Consolidation epochs")
                mlp_widths=gr.Textbox("64,64",label="MLP widths")
                mamba_hidden=gr.Number(16,precision=0,label="Mamba hidden size"); mamba_layers=gr.Number(1,precision=0,label="Mamba layers")
                mamba_ff=gr.Number(64,precision=0,label="Mamba feed-forward size")
                rbf_centers=gr.Number(32,precision=0,label="RBF centers"); fourier_frequencies=gr.Number(32,precision=0,label="Fourier frequencies")
        with gr.Tab("Extreme events"):
            extreme_mode=gr.Radio([("None","none"),("Statistical","statistical"),("Structural — coming soon","structural")],value="none",label="Mode")
            definition=gr.Radio([("Automatic per-basin Q95","automatic_q95"),("Manual absolute discharge","absolute"),("Manual quantile","quantile")],value="automatic_q95",label="Definition")
            extreme_value=gr.Number(value=.95,label="Global value")
            gr.Markdown("Per-basin threshold overrides are available through the Python service API.")
        with gr.Tab("Results"):
            run=gr.Button("Run experiment",variant="primary"); cancel=gr.Button("Cancel")
            status=gr.Markdown()
            with gr.Row():
                historical_map=map_component.gradio_component(
                    gr, interaction="activate", label="Historical / training status"
                )
                future_map=map_component.gradio_component(
                    gr, future=True, interaction="activate", label="Prediction targets"
                )
            active_basin=gr.Dropdown(choices=choices,label="Active result basin")
            with gr.Row(): hydrograph=gr.Plot(label="Observed and predicted hydrograph"); scatter=gr.Plot(label="Observed vs predicted")
            with gr.Row(): routing_plot=gr.Plot(label="Complex-expert usage"); loss_plot=gr.Plot(label="Loss traces")
            metrics=gr.JSON(label="Metrics"); extremes=gr.Dataframe(label="Extreme-event results")
            with gr.Row(): predictions_file=gr.File(label="Predictions CSV"); metrics_file=gr.File(label="Metrics CSV"); config_file=gr.File(label="Resolved config"); archive_file=gr.File(label="Run archive")

        def selection_payload(value):
            selected = [str(item) for item in (value or ()) if str(item) in eligible]
            return selected, map_component.payload(
                selected, future=True, interaction="toggle"
            )

        select_all.click(
            lambda: selection_payload(catalog.eligible_ids), outputs=[targets, basin_map]
        )
        clear.click(lambda: selection_payload(()), outputs=[targets, basin_map])
        targets.change(
            lambda value: map_component.payload(
                value or (), future=True, interaction="toggle"
            ),
            inputs=targets, outputs=basin_map,
        )
        basin_map.input(
            lambda value: selection_payload(value.get("selected", ()) if isinstance(value, dict) else ())[0],
            inputs=basin_map, outputs=targets,
        )
        cancel.click(
            lambda token: (token.cancel() if token else None, "Cancellation requested.")[1],
            inputs=token_state, outputs=status, queue=False,
        )

        def result_views(result, basin):
            import plotly.graph_objects as go

            target_basins = tuple(result.resolved_config["basins"]["target_basins"])
            basin = str(basin) if basin is not None else (target_basins[0] if target_basins else None)
            if basin not in target_basins:
                basin = target_basins[0] if target_basins else None
            rows=[row for row in result.predictions if row.basin_id == basin]
            hydro=go.Figure()
            if rows:
                by_date={}
                for row in rows: by_date.setdefault(row.target_date, [[], row.observed_mm_day])[0].append(row.predicted_mm_day)
                days=sorted(by_date); predicted=[sum(by_date[x][0])/len(by_date[x][0]) for x in days]; observed=[by_date[x][1] for x in days]
                hydro.add_scatter(x=days,y=observed,name="Observed"); hydro.add_scatter(x=days,y=predicted,name="Predicted")
            points=go.Figure(); points.add_scatter(x=[x.observed_mm_day for x in rows],y=[x.predicted_mm_day for x in rows],mode="markers")
            route=go.Figure(); route.add_scatter(x=[x.target_date for x in rows],y=[x.complex_weight for x in rows],mode="lines",name="Complex weight")
            losses=go.Figure()
            for phase, trace in result.loss_traces.items(): losses.add_scatter(y=trace,mode="lines+markers",name=phase)
            training=result.resolved_config["effective_training_basins"]
            return (
                basin,
                map_component.payload(target_basins, active=basin, training=training,
                                      interaction="activate"),
                map_component.payload(target_basins, active=basin, future=True,
                                      interaction="activate"),
                hydro, points, route, losses,
            )

        def execute(target_values, scope_value, token, *values, progress=gr.Progress()):
            if values[28] == "structural":
                raise gr.Error("Structural extreme-event analysis is coming soon")
            try:
                config = _config_from_ui_values(data_root, target_values, scope_value, values)
                result=run_regional_experiment(
                    config, lambda fraction,message: progress(fraction,desc=message), token
                )
            except (ValueError, FileNotFoundError) as error:
                raise gr.Error(str(error)) from error
            paths=write_result_artifacts(result)
            message="Cancelled; showing completed partial results." if result.cancelled else f"Completed in {result.runtime_seconds:.1f}s"
            basin,historical,future,hydro,points,route,losses=result_views(
                result, (target_values or [None])[0]
            )
            metric_value={"aggregate":result.aggregate_metrics,"per_basin":result.per_basin_metrics,"routing":result.routing_diagnostics,"parameters":result.parameter_count,"runtime_seconds":result.runtime_seconds,"failures":result.failures}
            return token,result,message,historical,future,basin,hydro,points,route,losses,metric_value,result.extreme_events,paths["predictions"],paths["metrics"],paths["config"],paths["archive"]

        def refresh_result(basin, result):
            if result is None:
                return basin, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
            return result_views(result, basin)

        def activate_from_map(value, result):
            basin = value.get("active") if isinstance(value, dict) else None
            return refresh_result(basin, result)

        def refresh_result_views(basin, result):
            return refresh_result(basin, result)[1:]

        inputs=[targets,scope,train_start,train_end,val_start,val_end,test_start,test_end,approach,complex_expert,simple_expert,epochs,sequence,seeds,batch,learning_rate,noise,percentile,temperature,physics_weight,interface_weight,teacher_epochs,distill_epochs,consolidation_epochs,mlp_widths,mamba_hidden,mamba_layers,mamba_ff,rbf_centers,fourier_frequencies,extreme_mode,routing_weight,compute_weight,device,definition,extreme_value]
        prepare = run.click(lambda: CancellationToken(), outputs=token_state, queue=False)
        prepare.then(execute, inputs=[targets, scope, token_state, *inputs[2:]],
            outputs=[token_state,experiment_state,status,historical_map,future_map,active_basin,
                     hydrograph,scatter,routing_plot,loss_plot,metrics,extremes,
                     predictions_file,metrics_file,config_file,archive_file],
            concurrency_id="regional-training", concurrency_limit=1)
        result_outputs=[active_basin,historical_map,future_map,hydrograph,scatter,routing_plot,loss_plot]
        active_basin.change(
            refresh_result_views, inputs=[active_basin,experiment_state], outputs=result_outputs[1:]
        )
        historical_map.input(
            activate_from_map, inputs=[historical_map,experiment_state], outputs=result_outputs
        )
        future_map.input(
            activate_from_map, inputs=[future_map,experiment_state], outputs=result_outputs
        )
    return app.queue(default_concurrency_limit=1)


def main() -> None:
    parser=argparse.ArgumentParser(description="Launch the regional CAMELS-CH Gradio application")
    parser.add_argument("--data-root",type=Path,default=os.environ.get("CAMELS_CH_ROOT"))
    parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=7860)
    args=parser.parse_args()
    if args.data_root is None: parser.error("--data-root or CAMELS_CH_ROOT is required")
    CAMELSCHCatalog.validate_layout(args.data_root)
    build_app(args.data_root).launch(server_name=args.host,server_port=args.port,share=True)


if __name__ == "__main__": main()
