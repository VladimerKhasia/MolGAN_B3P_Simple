import streamlit as st
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from PIL import Image
import warnings
#import matplotlib.pyplot as plt
# import shutil 
# import deepchem as dc

from rdkit import Chem, RDLogger
from rdkit.Chem import PyMol, AllChem, Draw
#from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage

from stmol import showmol
import py3Dmol

sys.path.append("gun_model")
sys.path.append("mpnn_model")

@st.cache(allow_output_mutation=True)
def load_gun_model(): 
    gun_model = keras.models.load_model("gun_model")
    gun_model.summary()  # to make it visible when model is reloaded
    return gun_model

@st.cache(allow_output_mutation=True)
def load_mpnn_model(): 
    mpnn_model = keras.models.load_model("mpnn_model")
    mpnn_model.summary()  
    return mpnn_model

def generate():
    atom_mapping = {
        "C": 0,
        0: "C",
        "N": 1,
        1: "N",
        "O": 2,
        2: "O",
        "F": 3,
        3: "F",
    }

    bond_mapping = {
        "SINGLE": 0,
        0: Chem.BondType.SINGLE,
        "DOUBLE": 1,
        1: Chem.BondType.DOUBLE,
        "TRIPLE": 2,
        2: Chem.BondType.TRIPLE,
        "AROMATIC": 3,
        3: Chem.BondType.AROMATIC,
    }

    NUM_ATOMS = 9  
    ATOM_DIM = 4 + 1  
    BOND_DIM = 4 + 1  
    LATENT_DIM = 64  

    def graph_to_molecule(graph):
        adjacency, features = graph
        molecule = Chem.RWMol()
        keep_idx = np.where(
            (np.argmax(features, axis=1) != ATOM_DIM - 1)
            & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
        )[0]
        features = features[keep_idx]
        adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

        for atom_type_idx in np.argmax(features, axis=1):
            atom = Chem.Atom(atom_mapping[atom_type_idx])
            molecule.AddAtom(atom)

        (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
        for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
            if atom_i == atom_j or bond_ij == BOND_DIM - 1:
                continue
            bond_type = bond_mapping[bond_ij]
            molecule.AddBond(int(atom_i), int(atom_j), bond_type)

        flag = Chem.SanitizeMol(molecule, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            return None

        return molecule     

    def sample(generator, batch_size):
        z = tf.random.normal((batch_size, LATENT_DIM))
        graph = generator.predict(z)
        adjacency = tf.argmax(graph[0], axis=1)
        adjacency = tf.one_hot(adjacency, depth=BOND_DIM, axis=1)
        adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
        features = tf.argmax(graph[1], axis=2)
        features = tf.one_hot(features, depth=ATOM_DIM, axis=2)
        return [
            graph_to_molecule([adjacency[i].numpy(), features[i].numpy()])
            for i in range(batch_size)
        ]

    gun_model= load_gun_model()
    generated_molecules_raw = sample(gun_model, batch_size=48)
    generated_molecules = [m for m in generated_molecules_raw if m is not None]
    
    return generated_molecules


def classify_generated():
  class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


  class AtomFeaturizer(Featurizer):
      def __init__(self, allowable_sets):
          super().__init__(allowable_sets)

      def symbol(self, atom):
          return atom.GetSymbol()

      def n_valence(self, atom):
          return atom.GetTotalValence()

      def n_hydrogens(self, atom):
          return atom.GetTotalNumHs()

      def hybridization(self, atom):
          return atom.GetHybridization().name.lower()


  class BondFeaturizer(Featurizer):
      def __init__(self, allowable_sets):
          super().__init__(allowable_sets)
          self.dim += 1

      def encode(self, bond):
          output = np.zeros((self.dim,))
          if bond is None:
              output[-1] = 1.0
              return output
          output = super().encode(bond)
          return output

      def bond_type(self, bond):
          return bond.GetBondType().name.lower()

      def conjugated(self, bond):
          return bond.GetIsConjugated()

  atom_featurizer = AtomFeaturizer(
      allowable_sets={
          "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
          "n_valence": {0, 1, 2, 3, 4, 5, 6},
          "n_hydrogens": {0, 1, 2, 3, 4},
          "hybridization": {"s", "sp", "sp2", "sp3"},
      }
  )

  bond_featurizer = BondFeaturizer(
      allowable_sets={
          "bond_type": {"single", "double", "triple", "aromatic"},
          "conjugated": {True, False},
      }
  )

  def prepare_batch(x_batch, y_batch):
      atom_features, bond_features, pair_indices = x_batch
      num_atoms = atom_features.row_lengths()
      num_bonds = bond_features.row_lengths()
      molecule_indices = tf.range(len(num_atoms))
      molecule_indicator = tf.repeat(molecule_indices, num_atoms)
      gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
      increment = tf.cumsum(num_atoms[:-1])
      increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
      pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
      pair_indices = pair_indices + increment[:, tf.newaxis]
      atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
      bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

      return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch

  def graph_from_molecule(molecule):

      atom_features = []
      bond_features = []
      pair_indices = []

      for atom in molecule.GetAtoms():
          atom_features.append(atom_featurizer.encode(atom))

          # self-loops
          pair_indices.append([atom.GetIdx(), atom.GetIdx()])
          bond_features.append(bond_featurizer.encode(None))

          for neighbor in atom.GetNeighbors():
              bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
              pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
              bond_features.append(bond_featurizer.encode(bond))

      return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


  def graphs_from_molecules(generated_molecules):

      atom_features_list = []
      bond_features_list = []
      pair_indices_list = []

      for molecule in generated_molecules:
          atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

          atom_features_list.append(atom_features)
          bond_features_list.append(bond_features)
          pair_indices_list.append(pair_indices)

      return (
          tf.ragged.constant(atom_features_list, dtype=tf.float32),
          tf.ragged.constant(bond_features_list, dtype=tf.float32),
          tf.ragged.constant(pair_indices_list, dtype=tf.int64),
      )

  def MPNNDataset(X, y, batch_size=32, shuffle=False):
      dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
      if shuffle:
          dataset = dataset.shuffle(1024)
      return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)


  mpnn_model = load_mpnn_model()

  generated_molecules = generate()

  X = graphs_from_molecules(generated_molecules)
  y = pd.Series([0]*len(generated_molecules)) 
  bbbp = MPNNDataset(X, y)

  y_pred = tf.squeeze(mpnn_model.predict(bbbp), axis=1)

  legends = [f"{100 * y_pred[i]:.0f}% likelihood" for i in range(len(y_pred))]
  #MolsToGridImage(generated_molecules[:25], molsPerRow=4, legends=legends)  #subImgSize=(150, 150) 

  return generated_molecules, legends 


def rdkit_3Dmol(molecule):
  mol = Chem.AddHs(m)
  AllChem.EmbedMolecule(mol)
  modifiedmol = Chem.MolToMolBlock(mol)
  return modifiedmol
      
def render(modifiedmol, bcolor, style):
    xyzview = py3Dmol.view() 
    xyzview.addModel(modifiedmol,'mol')
    xyzview.setStyle({style:{}})
    xyzview.setBackgroundColor(bcolor)
    xyzview.zoomTo()
    showmol(xyzview,height=500,width=800)


if __name__ == '__main__':
    seed = 27834096
    st.sidebar.title("V. Khasia's Simple Demo")
    st.sidebar.write(
        """Generate molecules and define likelihood of their hematoencephalic barrier permeability.
        """
    )
    
    bcolor = st.sidebar.color_picker('Pick A Color', '#000000')
    style = st.sidebar.selectbox('style',['stick','sphere','line','cross']) # others: 'cartoon', 'clicksphere'
    
    button = st.sidebar.button('generate molecules') 
    st.sidebar.title("")
    st.sidebar.text("")

    #st.sidebar.markdown("***")
    #st.sidebar.caption(f"Streamlit version `{st.__version__}`")

    if button:
      molecules, legends = classify_generated()
      molecules = molecules[:15]
      legends = legends[:15]
      #m = [Draw.MolToImage(m) for m in molecules[:25]] 
      #m = MolsToGridImage(mols[:25], molsPerRow=4, subImgSize=(150, 150), highlightAtomLists=None, useSVG=False, returnPNG=True)
      st.title("""Generated molecules with their BBBP likelihood""")
      #st.image(m, caption=legends, use_column_width='auto')
      
      for i, m in enumerate(molecules):
        legend = legends[i]
        st.write(legend)
        mol = rdkit_3Dmol(m)
        render(mol, bcolor, style)
        if i == legend:
          break 

    else:
      st.title("""Generate small realistic molecules !""") 
      st.text("")
      with st.expander("See explanation"):
        st.write("""
            Hi,
            this demo app is based on two deep neural networks and it is created for my free tutorials 📚
            The generator of molecules is simplified implementation of plane graph convolution network.
            Blood brain barrier permeability is assessed by simplified implementation of message passing deep neural network.
            Code will be available on one of my 
            [github](https://github.com/VladimerKhasia) accounts 🧵

            In the sidebar you will find: 
            🌈 Colorpicker, for selecting background color for the generated molecules 
            🌺 Style selector, which allows you to choose style for molecules that will be generated
            🍄 'generate molecules' button explains itself, by clicking on it you generate 15 molecules with
            likelihoods on Blood brain barrier permeability 💊 (so, do not forget to scroll down when generated)

            Generated molecules are interactive! You can rotate them any direction (move mouse by holding left click on molecule)
            You can also zoom in or out (by scrolling with mouse after clicking on molecule).
            
            From the menu icon in the upper right corner you can choose settings and change theme to dark or light etc.

            hope you enjoy!✨
            Vladimer Khasia
        """)

      # prot_str='1A2C,1BML,1D5M,1D5X,1D5Z,1D6E,1DEE,1E9F,1FC2,1FCC,1G4U,1GZS,1HE1,1HEZ,1HQR,1HXY,1IBX,1JBU,1JWM,1JWS'
      # prot_list=prot_str.split(',')
      # bcolor = st.sidebar.color_picker('Pick A Color', '#000000')
      # protein=st.sidebar.selectbox('select protein',prot_list)
      # style = st.sidebar.selectbox('style',['cartoon','line','cross','stick','sphere','clicksphere'])

      # xyzview = py3Dmol.view(query='pdb:'+protein)
      # xyzview.setStyle({style:{'color':'spectrum'}})
      # xyzview.setBackgroundColor(bcolor)
      # showmol(xyzview, height = 500,width=800)

      #https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-
      
      # R E F F E R E N C E : 
      # 1. https://arxiv.org/abs/1704.01212 and https://arxiv.org/abs/1805.11973 
      # 2. https://deepchem.readthedocs.io/en/latest/api_reference/models.html#mpnnmodel , https://deepchem.readthedocs.io/en/latest/api_reference/models.html#basicmolganmod
      # 3. Alexander Kensert - https://github.com/keras-team/keras-io/blob/master/examples/graph/mpnn-molecular-graphs.py