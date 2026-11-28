"""Supplied motor data stays separate, finite, fixed, and available to consumers."""
import copy
import math

import mujoco
import numpy as np
import pytest
import yaml

from draft.editor.core.session import MotorInputs
from draft.generation.generator import RobotGenerator
from draft.generation.wheel_leg_inspection import inspect_sitting
from draft.robots import robot_dir
from draft.trends.motor_catalog import find_by_name, load_catalog, supplied_entry
from draft.trends.motor_solve import design_class, envelope_curve


@pytest.fixture(scope='module')
def params():
    return RobotGenerator(robot_dir('galileo_x')).params


def test_local_profiles_do_not_change_the_reference_population(params):
    assert len(load_catalog()) == 114
    assert find_by_name('JM150') is None
    assert find_by_name('M150', params)['name'] == 'JM150'
    assert find_by_name('XQ-JE-250N-25-2-CN_48', params)['name'] == 'JM250'


@pytest.mark.parametrize('cls,name,continuous,peak10,peak3,rpm,mass,diameter,length', [
    ('J1','JM150',45,150,210,150.6,0.8,75,80),
    ('J2','JM100',30,100,140,159.3,0.59,75,55),
    ('J3','JM250',80,250,350,145.,1.15,90,80),
])
def test_supplier_dimensions_limits_and_digitized_curves_are_preserved(
        params, cls,name,continuous,peak10,peak3,rpm,mass,diameter,length):
    design = design_class(cls, params)
    spec = design.sized['specification']
    assert design.fixed_spec
    assert spec['continuous_torque'] == continuous
    assert spec['peak_torque_10s'] == peak10
    assert spec['peak_torque_3s'] == peak3
    assert design.mass == mass
    assert (design.sized['r'],design.sized['L']) == pytest.approx((diameter/2000,length/1000))
    curve = spec['performance_curve']
    assert len(curve['ibus_a']) == len(curve['efficiency_pct']) == 11
    assert spec['source_sha256'] and len(spec['source_sha256']) == 64
    assert spec['torque_speed_curve'][0] == pytest.approx([0,rpm*math.pi/30])
    speeds = [w for t,w in spec['torque_speed_curve']]
    np.testing.assert_allclose(envelope_curve(design.point, speeds), curve['torque_nm'])
    assert math.isnan(envelope_curve(design.point, [0])[0])  # unmeasured stall region
    assert design.point.power == pytest.approx(max(t*w for t,w in spec['torque_speed_curve']))
    assert design.densities()['torque_density']['status'] == 'supplier_spec'
    assert not design.errors


@pytest.mark.parametrize('override', [{'J1_motor_effort': 210}, {'J1_motor_gear': 31},
                                    {'J1_motor_aspect': 0.5}, {'J1_motor_mode':'power_gear'}])
def test_fixed_supplier_hardware_cannot_be_resized_or_promoted_to_the_3s_peak(params, override):
    with pytest.raises(ValueError, match='fixed supplier'):
        design_class('J1', {**params, **override, 'allow_hypothetical':True})


def test_editor_selecting_another_supplied_motor_reseeds_physical_inputs(params):
    inputs = MotorInputs(('J1',), params)
    inputs.set_catalog_name('J1','JM100')
    design = inputs.design('J1')
    assert design.mass == .590
    assert design.point.tau == 100
    assert design.point.gear == 36
    assert inputs.collect()['_motor_catalog']


@pytest.mark.parametrize('mutation,message', [
    ('length','matching lengths'),('speed','monotonic'),('efficiency','efficiency'),
    ('mass','positive'),('peak','ordering'),
])
def test_invalid_supplied_data_is_rejected(params, mutation, message):
    spec = copy.deepcopy(params['_motor_catalog']['JM150'])
    if mutation=='length':spec['performance_curve']['ibus_a'].pop()
    elif mutation=='speed':spec['performance_curve']['speed_rpm'][2]=200
    elif mutation=='efficiency':spec['performance_curve']['efficiency_pct'][3]=101
    elif mutation=='mass':spec['mass']=-1
    elif mutation=='peak':spec['peak_torque_3s']=120
    with pytest.raises(ValueError,match=message):supplied_entry(spec)


def test_generated_curve_metadata_and_centerline_outboard_layout(tmp_path):
    RobotGenerator(robot_dir('galileo_x')).generate(tmp_path)
    model=mujoco.MjModel.from_xml_path(str(tmp_path/'galileo_x.xml'))
    params=yaml.safe_load((tmp_path/'parameters_resolved.yaml').read_text(encoding='utf-8'))
    assert params['torso_lz']==.2
    assert params['hip_abduction_link_length']==pytest.approx(.09)
    for leg,sy in [('fl',1),('fr',-1),('rl',1),('rr',-1)]:
        for j,cls,continuous,peak3 in [(1,'J1',45,210),(2,'J2',30,140),(3,'J3',80,350)]:
            curve=model.numeric(f'{leg}_j{j}_torque_speed_curve').data.reshape(-1,2)
            np.testing.assert_allclose(curve,params[f'{cls}_motor_torque_speed_curve'],rtol=5e-6)
            assert model.numeric(f'{leg}_j{j}_continuous_torque').data[0]==continuous
            assert model.numeric(f'{leg}_j{j}_peak_torque_3s').data[0]==peak3
        data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,model.key('standing').id)
        mujoco.mj_forward(model,data)
        hip=data.geom_xpos[model.geom(f'{leg}_j1_housing_motor').id]
        assert hip[2]==data.xpos[model.body('torso_link').id,2]
        shank=data.xpos[model.body(f'{leg}_shank_link').id]
        wheel=data.xanchor[model.joint(f'{leg}_j4').id]
        assert sy*(wheel[1]-shank[1])>0
    result=inspect_sitting(model,params)
    assert result.report['minimum']['parts']['distance_m']>=.005-2e-6


def test_old_abduction_spacing_exposes_jm250_torso_interference(tmp_path):
    RobotGenerator(robot_dir('galileo_x')).generate(tmp_path,params_override={'hip_abduction_link_length':.0825})
    model=mujoco.MjModel.from_xml_path(str(tmp_path/'galileo_x.xml'))
    params=yaml.safe_load((tmp_path/'parameters_resolved.yaml').read_text(encoding='utf-8'))
    with pytest.raises(ValueError,match='parts:.*torso_link_mesh'):
        inspect_sitting(model,params)
