import io
import os
import glob
import random
import subprocess
import tempfile
from ase import Atoms
from ase.io import read, write
from parsl import python_app, bash_app
from alframework.tools.tools import system_checker
from alframework.tools.tools import load_module_from_config
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator

def qm9_build(start_system, molecule_library_dir):

    #ensure system adhears to formating convention
    if type(start_system) == list:
        for system in start_system: 
            system_checker(system)
        prefix = start_system[0][0]['moleculeid']
    else: 
        system_checker(start_system)
        prefix = system[0]['moleculeid']
 
    #path to xyz
    xyzs = sorted(glob.glob(os.path.join(molecule_library_dir, '*.xyz')))
    if type(start_system) == list:
        xyz = [random.choice(xyzs) for _ in range(len(start_system))]
    else:
        xyz = random.choice(xyzs)
    
    #write structures
    if type(start_system) == list:
        for n in range(len(start_system)):
            with open(xyz[n], 'r') as f:
                data = [line for line in f.readlines()]
            natoms = int(data[0].strip())
            xyzfile = io.StringIO("".join(data[:natoms+2])) 
            start_system[n][1] = read(xyzfile, format='xyz', index=0)
    else:
        with open(xyz, 'r') as f:
            data = [line for line in f.readlines()]
        natoms = int(data[0].strip())
        xyzfile = io.StringIO("".join(data[:natoms+2])) 
        start_system[1] = read(xyzfile, format='xyz', index=0)

    return(start_system)


def qm9_metad(start_system, ase_calculator, xtb_command='xtb', hmass=2, time=50., temp=400., step=0.5, shake=0,
                dump=100, save=100, kpush=0.05, alp=1.0, Escut=20., Fscut=0.3, store_dir=None):

    #ensure system adhears to formating convention
    system_checker(start_system)
    curr_sys = start_system[1]
    prefix = start_system[0]['moleculeid']

    if store_dir is not None:
        os.makedirs(store_dir, exist_ok=True)

    tmpdirname = os.path.join('/nfsscratch', 'alf', prefix)
    os.makedirs(tmpdirname, exist_ok=True)

    os.chdir(tmpdirname)
    #write input file
    with open('metadyn.inp', 'w') as f:
        #md block
        f.write(f"$md\n  hmass={hmass}\n  time={time}\n  temp={temp}\n  ")
        f.write(f"step={step}\n  shake={shake}\n  dump={dump}\n  $end\n")
        #metad block
        f.write(f"$metadyn\n  save={save}\n  kpush={kpush}\n  alp={alp}\n$end\n")
    
    #input coordinates
    write('input.xyz', curr_sys)

    #run metadynamics
    runcmd = [xtb_command, '--md', '--input', 'metadyn.inp', 'input.xyz']
    proc = subprocess.run(runcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if store_dir is not None:
        outfile = os.path.join(store_dir, f"{prefix}_sample.out")
        with open(outfile, 'w') as f:
            f.write(proc.stdout)
            f.write(proc.stderr)

    #read structures from trajectory
    cluster_xyz = os.path.join(tmpdirname, 'xtb.trj') 

    if store_dir is not None:
        outfile = os.path.join(store_dir, f"{prefix}_trj.xyz")
        runcmd = ['cp', cluster_xyz, outfile]
        _ = subprocess.run(runcmd)

    #loop over snapshots and return the snapshot if it meets criteria
    traj = read(cluster_xyz, format='xyz', index=':')
    for atoms in traj:
        atoms.set_calculator(ase_calculator)
        atoms.calc.calculate(atoms, properties=['energy_stdev','forces_stdev_mean','forces_stdev_max'])
        Es = atoms.calc.results['energy_stdev']
        Fs = atoms.calc.results['forces_stdev_mean']
        Fsmax = atoms.calc.results['forces_stdev_max']

        Ecrit = Es > float(Escut)
        Fcrit = Fs > float(Fscut)
        Fmcrit = Fsmax > 3*float(Fscut)

        if Ecrit or Fcrit or Fmcrit:
            start_system[1] = Atoms(atoms.get_chemical_symbols(), positions=atoms.get_positions(wrap=True),
                                    cell=atoms.get_cell(), pbc=atoms.get_pbc())
            start_system[2] = {}
            return(start_system)
        
    #if we get here, no snapshots met the criteria
    start_system[1] = Atoms()
    start_system[2] = {}
    return(start_system)

@python_app(executors=['alf_builder_executor'])
def qm9_build_task(moleculeids, builder_config):
    """
    Elements in builder params
        molecule_library_dir: path to library of molecular fragments to read in
    """
    if type(moleculeids) == list:
        empty_systems = [[{'moleculeid':moleculeid}, Atoms(), {}] for moleculeid in moleculeids]
    else:
        empty_systems = [{'moleculeid':moleculeids}, Atoms(), {}]
    system = qm9_build(empty_systems, **builder_config)
    if type(system) == list:
        for s in system:
            system_checker(s)
    else:
        system_checker(system)
    return(system)


@python_app(executors=['alf_sampler_executor'])
def qm9_metad_task(molecule_object, sampler_config, model_path, current_model_id, gpus_per_node):
    """
    Elements in sampler params
        xtb_command: path to xTB
        hmass: mass of hydrogen atoms (amu)
        time: integration time (ps)
        temp: temperature (K)
        step: step size (fs)
        shake: bond constraints (0=off, 1=X-H-bonds, 2=all-bonds)
        dump: trajectory write interval (fs)
        save: max number of structures for RMSD collective variable
        kpush: scaling factor for Gaussian potential used in RMSD CV
        alp: width of gaussian potential used in RMSD CV
        Escut: Energy standard deviation threshold for capturing frame
        Fscut: Force standard deviation threshold for capturing frame
        store_dir: optional path to storage directory
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(os.environ.get('PARSL_WORKER_RANK'))%gpus_per_node)
    system_checker(molecule_object)
    calc_class = load_module_from_config(sampler_config, 'ase_calculator')
    calculator_list = calc_class(model_path.format(current_model_id) + '/',device='cuda:0')
    ase_calculator = MLMD_calculator(calculator_list, **sampler_config['MLMD_calculator_options'])
    feed_parameters = {**sampler_config}
    _ = feed_parameters.pop('ase_calculator')
    _ = feed_parameters.pop('MLMD_calculator_options')
    system = qm9_metad(molecule_object, ase_calculator, **feed_parameters)
    system_checker(system)
    return(system)
