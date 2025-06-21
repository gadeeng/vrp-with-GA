import streamlit as st
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from datetime import datetime

# Load data 
st.title("🧬 Vehicle Routing Problem with Genetic Algorithm")

st.markdown("### Upload File Excel")
uploaded_distance = st.file_uploader("Upload file Jarak antar lokasi (.xlsx)", type="xlsx")
uploaded_demand = st.file_uploader("Upload file Demand tiap lokasi (.xlsx)", type="xlsx")

with st.expander("📄 Lihat Contoh Format File yang Diupload"):
    st.markdown("**1. `distance_indomaret.xlsx`** (Sheet1):")
    st.markdown("""
    - Kolom pertama: Nama lokasi (boleh diabaikan)
    - Kolom-kolom selanjutnya: Matriks jarak antar lokasi (termasuk ke depot)
    - Format persegi (NxN), urutan baris dan kolom sama
    """)

    st.markdown("**2. `indomaret_demand.xlsx`** (Sheet1):")
    st.markdown("""
    - `nama_indomaret`: Nama lokasi
    - `demand`: Jumlah permintaan pada lokasi tersebut
    - **Catatan:** Lokasi pertama dianggap sebagai **depot** dan demand-nya bisa 0
    """)

    col1, col2 = st.columns(2)
    with col1:
        with open("data/distance_indomaret.xlsx", "rb") as f:
            st.download_button("⬇️ Download Contoh distance_indomaret.xlsx", f, "distance_indomaret.xlsx")

    with col2:
        with open("data/indomaret_demand.xlsx", "rb") as f:
            st.download_button("⬇️ Download Contoh indomaret_demand.xlsx", f, "indomaret_demand.xlsx")

def create_initial_population(num_locations, population_size):
    population = []
    for _ in range(population_size):
        chromosome = list(range(1, num_locations))  # 0 is the Depot
        random.shuffle(chromosome)
        population.append(chromosome)
    return population

def calculate_route_distance(route, distance_matrix):
    if not route:
        return 0
    distance = distance_matrix[0][route[0]]  # from depot to first location
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
    distance += distance_matrix[route[-1]][0]  # return to depot
    return distance

def split_route(chromosome, demands, vehicle_capacity):
    routes = []
    current_route = []
    current_load = 0
    for gene in chromosome:
        demand = demands[gene]
        if current_load + demand <= vehicle_capacity:
            current_route.append(gene)
            current_load += demand
        else:
            routes.append(current_route)
            current_route = [gene]
            current_load = demand
    if current_route:
        routes.append(current_route)
    return routes

def calculate_fitness(chromosome, distance_matrix, demands, vehicle_capacity):
    routes = split_route(chromosome, demands, vehicle_capacity)
    total_distance = sum(calculate_route_distance(route, distance_matrix) for route in routes)
    return 1 / total_distance if total_distance > 0 else float('inf')

def elite_selection(population, fitnesses):
    max_index = np.argmax(fitnesses)
    return population[max_index], fitnesses[max_index], max_index

def calculate_relative_and_cumulative_fitness(fitnesses, elite_index):
    total_fitness = sum(fitnesses) - fitnesses[elite_index]
    relative_fitness = [(fit / total_fitness if i != elite_index else 0) for i, fit in enumerate(fitnesses)]
    cumulative_fitness = np.cumsum(relative_fitness).tolist()
    return relative_fitness, cumulative_fitness

def roulette_selection(population, cumulative_fitness, elite_index):
    r = random.random()
    for i, cf in enumerate(cumulative_fitness):
        if r <= cf:
            return deepcopy(population[i])
    return deepcopy(population[-1])

def cycle_crossover(parent1, parent2, elite_index):
    size = len(parent1)
    child = [None] * size
    index = 0
    while child[index] is None:
        child[index] = parent1[index]
        index = parent1.index(parent2[index])
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return child

def inversion_mutation(chromosome, elite_index):
    chrom = deepcopy(chromosome)
    idx1, idx2 = sorted(random.sample(range(len(chrom)), 2))
    chrom[idx1:idx2] = reversed(chrom[idx1:idx2])
    return chrom

if uploaded_distance and uploaded_demand:
    try:
        distance_df = pd.read_excel(uploaded_distance)
        demand_df = pd.read_excel(uploaded_demand)

        distance_matrix = distance_df.iloc[:, 1:].to_numpy()
        demands = demand_df['demand'].to_numpy()
        locations = demand_df['nama_indomaret'].to_list()

        # Input parameter
        st.sidebar.markdown("## Parameter Algoritma Genetika")
        VEHICLE_CAPACITY = st.sidebar.number_input("Kapasitas Kendaraan", min_value=1, value=500)
        POP_SIZE = st.sidebar.number_input("Ukuran Populasi", min_value=2, value=10)
        CROSSOVER_RATE = st.sidebar.slider("Tingkat Crossover", 0.0, 1.0, 0.7)
        MUTATION_RATE = st.sidebar.slider("Tingkat Mutasi", 0.0, 1.0, 0.1)
        MAX_GENERATIONS = st.sidebar.number_input("Jumlah Generasi Maksimum", min_value=1, value=10)

        if st.button("Run"):
            with st.spinner("Sedang mencari solusi optimal..."):
                def genetic_algorithm_local():
                    num_locations = len(locations)
                    population = create_initial_population(num_locations, POP_SIZE)
                    best_solution = None
                    best_fitness = -float('inf')

                    for _ in range(MAX_GENERATIONS):
                        fitnesses = [calculate_fitness(route, distance_matrix, demands, VEHICLE_CAPACITY) for route in population]
                        max_fitness_idx = np.argmax(fitnesses)
                        if fitnesses[max_fitness_idx] > best_fitness:
                            best_fitness = fitnesses[max_fitness_idx]
                            best_solution = deepcopy(population[max_fitness_idx])

                        elite_chrom, elite_fitness, elite_index = elite_selection(population, fitnesses)
                        relative_fitness, cumulative_fitness = calculate_relative_and_cumulative_fitness(fitnesses, elite_index)

                        new_population = [elite_chrom]
                        crossover_count = int((POP_SIZE - 1) * CROSSOVER_RATE)
                        if crossover_count % 2 == 1:
                            crossover_count -= 1
                        for _ in range(crossover_count // 2):
                            parent1 = roulette_selection(population, cumulative_fitness, 0)
                            parent2 = roulette_selection(population, cumulative_fitness, 0)
                            child1 = cycle_crossover(parent1, parent2, 0)
                            child2 = cycle_crossover(parent2, parent1, 0)
                            new_population.extend([child1, child2])
                        mutation_count = int((POP_SIZE - len(new_population)) * MUTATION_RATE)
                        for _ in range(mutation_count):
                            parent = roulette_selection(population, cumulative_fitness, 0)
                            child = inversion_mutation(parent, 0)
                            new_population.append(child)
                        while len(new_population) < POP_SIZE:
                            selected = roulette_selection(population, cumulative_fitness, 0)
                            new_population.append(selected)
                        population = new_population

                    vehicle_routes = split_route(best_solution, demands, VEHICLE_CAPACITY)
                    total_distance = sum(calculate_route_distance(route, distance_matrix) for route in vehicle_routes)
                    return vehicle_routes, total_distance

                routes, total_distance = genetic_algorithm_local()

                st.success("Solusi ditemukan!")
                st.markdown("### Hasil Rute Terbaik")
                for i, route in enumerate(routes):
                    route_names = [locations[0]] + [locations[loc] for loc in route] + [locations[0]]
                    route_load = sum(demands[loc] for loc in route)
                    route_dist = calculate_route_distance(route, distance_matrix)
                    st.markdown(f"**Kendaraan {i+1}**")
                    st.markdown(f"- Rute: {' → '.join(route_names)}")
                    st.markdown(f"- Beban: {route_load}/{VEHICLE_CAPACITY}")
                    st.markdown(f"- Jarak: {route_dist:.2f} km")
                st.markdown(f"### Total Jarak Seluruh Rute: `{total_distance:.2f} km`")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file atau menjalankan algoritma: {e}")
else:
    st.info("Silakan upload kedua file terlebih dahulu.")
