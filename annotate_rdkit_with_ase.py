import os
import string
import random
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Geometry.rdGeometry import Point3D
from ase import Atoms
from ase.optimize import BFGS

def rdkit2ase(mol_rdkit, conformation_index):
    '''Convert an RDKit molecule to an ASE Atoms object'''
    elements = [atom.GetSymbol() for atom in mol_rdkit.GetAtoms()]
    positions = mol_rdkit.GetConformer(conformation_index).GetPositions()
    mol_ase = Atoms(elements, positions)
    return mol_ase

def annotate_molecule_property(property_name, property_function, ase_calculator, mol_rdkit,
        conformation_index = 0):
    '''Given a property function (ASE to number), and an ASE calculator, and an
    RDKit molecule, calculate the property with the given calculator, and add
    it to the RDKit molecule as a molecule property, with the given name'''
    ase_molecule = rdkit2ase(mol_rdkit, conformation_index)
    ase_molecule.calc = ase_calculator
    property_value = property_function(ase_molecule)
    mol_rdkit.SetDoubleProp(property_name, property_value)

def annotate_atom_property(property_name, property_function, ase_calculator, mol_rdkit,
        conformation_index = 0):
    '''Given an atom property function (ASE to iterable of numbers, one for
    each atom, in order), and an ASE calculator, and an RDKit molecule,
    annotate the RDKit molecule with the atom property'''
    ase_molecule = rdkit2ase(mol_rdkit, conformation_index)
    ase_molecule.calc = ase_calculator
    property_values = property_function(ase_molecule)
    for atom, property_value in zip(mol_rdkit.GetAtoms(), property_values):
        atom.SetDoubleProp(property_name, property_value)

def optimize_geometry(ase_calculator, mol_rdkit, conformation_index = None):
    '''Given an ASE calculator and an RDKit molecule, optimize the geometry
    using that calculator'''
    
    if conformation_index is None:
        # Generate initial conformer
        mol_rdkit.RemoveAllConformers()
        conformation_index = EmbedMolecule(mol_rdkit)

    if conformation_index != -1:
        # Create a random sequence of characters for the temp files. This way, jobs
        # running on different threads don't try to access the same file
        noise = "".join(random.choices(string.ascii_uppercase + string.digits, k=20))
        # Name of the trajectory file
        traj_filename = f"tmp_opt_{noise}.traj"

        # Optimize the geometry
        mol_opt_ase = rdkit2ase(mol_rdkit, conformation_index)
        mol_opt_ase.calc = ase_calculator
        opt = BFGS(mol_opt_ase, trajectory = traj_filename, logfile = None)
        opt.run(fmax=0.05)
        os.remove(traj_filename)

        # Set the optimized geometry as the conformer
        positions = mol_opt_ase.get_positions()
        target_conformer = mol_rdkit.GetConformer(conformation_index)
        for i, row in enumerate(positions):
            x = row[0]
            y = row[1]
            z = row[2]
            rdkit_point = Point3D(x, y, z)
            target_conformer.SetAtomPosition(i, rdkit_point)
    
    else:
        raise ValueError("Failed to generate conformation")

    return conformation_index
